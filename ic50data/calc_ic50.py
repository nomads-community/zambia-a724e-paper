import os
import glob
import string
import numpy as np
import pandas as pd

from scipy.optimize import curve_fit

from itertools import product
from dataclasses import dataclass
from tabulate import tabulate

import seaborn as sns
import matplotlib.pyplot as plt

# Settings
DIR_INPUT = "raw_data"
DIR_OUTPUT = "results"


# --------------------------------------------------------------------------------
# ASSAY SETUP
#
# --------------------------------------------------------------------------------


@dataclass
class Plate:
    n_row: int = 8
    n_col: int = 12

    @property
    def rows(self) -> list[str]:
        return list(string.ascii_uppercase[: self.n_row])

    @property
    def cols(self) -> list[int]:
        return list(range(1, self.n_col + 1))

    @property
    def wells_by_rows(self) -> list[str]:
        return [f"{r}{c}" for r, c in product(self.rows, self.cols)]

    @property
    def wells_by_cols(self) -> list[str]:
        return [f"{r}{c}" for c, r in product(self.cols, self.rows)]


PLATE = Plate()


def make_serial_dilution(
    high_nm: float, fac: float, n: int = 8, set_zero: bool = True
) -> list[float]:
    """Make a serial dilution"""
    v = [high_nm / fac**i for i in range(0, n)]
    if set_zero:
        v[-1] = 0
    return v


def get_plate_layout(samples: list[str] = None, dilution_factor: int = 3) -> pd.DataFrame:
    """
    This effectively hardcodes the layout of the late, i.e.
    how the drug serial dilutions and samples are arranged
    NB: this is fragile, will break e.g. if not the right number of samples.
    """
    n_samps = len(samples)
    n_reps = 2
    n_drugs = 2
    N = PLATE.n_row * n_samps * n_reps * n_drugs
    if samples is None:
        samples = [f"samp{i}" for i in range(1, 4)]
    assert len(samples) == n_samps
    drug_dils = {
        "dha": make_serial_dilution(250, dilution_factor),
        "lum": make_serial_dilution(1000, dilution_factor),
    }
    return pd.DataFrame(
        {
            "well": PLATE.wells_by_cols[:N],
            "sample_id": [
                s for s in samples for _ in range(n_reps * len(drug_dils) * PLATE.n_row)
            ],
            "drug": (["dha"] * PLATE.n_row * n_reps + ["lum"] * PLATE.n_row * n_reps)
            * n_samps,
            "conc_nm": (drug_dils["dha"] * n_reps + drug_dils["lum"] * n_reps)
            * n_samps,
            "replicate": [
                i for i in range(1, n_reps + 1) for _ in range(0, PLATE.n_row)
            ]
            * (n_samps * n_drugs),
        }
    )


# --------------------------------------------------------------------------------
# LOADING AND PROCESSING FLUORESCENCE DATA
#
# --------------------------------------------------------------------------------

FOCUS_CYCLE = 15
RENAMER = {
    "SamplePos": "well",
    "SampleName": "sample_id",
    "Prog#": "prog",
    "Seg#": "seg",
    "Cycle#": "cycle",
    "Time": "time",
    "Temp": "temp",
    "465-510": "au",
}


def load_raw_data(input_txt: str) -> pd.DataFrame:
    return (
        pd.read_csv(input_txt, skiprows=1, sep="\t")
        .rename(RENAMER, axis=1)
        .query("cycle == @FOCUS_CYCLE")
    )


# --------------------------------------------------------------------------------
# IC50 ASSAY
#
# --------------------------------------------------------------------------------


def hill_equation(
    logx: np.ndarray[float], bottom: float, top: float, logIC50: float, hill: float
) -> np.ndarray[float]:
    return bottom + (top - bottom) / (1 + 10 ** ((logx - logIC50) * hill))


@dataclass
class IC50Result:
    sample_id: str
    drug: str
    ic50: float
    r2: float
    status: str
    note: str

@dataclass
class IC50ResultWithBootstrap:
    sample_id: str
    drug: str
    ic50: float
    r2: float
    ic50_lowci: float
    ic50_highci: float
    status: str
    note: str


class IC50Assay:
    def __init__(
        self,
        sample_id: str,
        drug: str,
        conc_nm: np.ndarray,
        fluor_vals: np.ndarray,
        bootstrap: bool = False,
    ) -> None:
        """Initialise an IC50 assay"""

        # Status indicators
        self.status = None
        self.note = None

        # Store inputs
        self.sample_id = sample_id
        self.drug = drug
        self.conc_nm = np.asarray(conc_nm, dtype=float)
        self.fluor_vals = np.asarray(fluor_vals, dtype=float)
        self.n_dilutions, self.n_reps = self.fluor_vals.shape

        # Sanity check
        self._check_inputs()

        # Compute percent survival
        self.fluor_dead = self.fluor_vals[
            0
        ]  # at maximum concentration, assume no growth, this is background
        self.fluor_untreated = self.fluor_vals[-1]
        self.percent_survival = 100 * self.fluor_vals / self.fluor_untreated
        self._check_survival_valid()

        # Fit the model
        self.conc_nm_log10 = np.log10(self.conc_nm[:-1])  # drop untreated
        self.percent_survival_fit = self.percent_survival[:-1]
        self.fit_params = self._fit_model(self.conc_nm_log10, self.percent_survival_fit)
        self.r2 = self._calc_r2(self.conc_nm_log10, self.percent_survival_fit, self.fit_params)

        # If it is passing, bootstrap
        self.bootstrap = bootstrap
        if self.bootstrap:
            self.ic50_lowci, self.ic50_highci = self._boostrap_ic50_ci()

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        assert len(df["sample_id"].unique()) == 1
        assert len(df["drug"].unique()) == 1
        return cls(
            sample_id=df["sample_id"].unique()[0],
            drug=df["drug"].unique()[0],
            conc_nm=np.array(
                [rdf["conc_nm"] for _, rdf in df.groupby("replicate")]
            ).transpose(),
            fluor_vals=np.array(
                [rdf["au"] for _, rdf in df.groupby("replicate")]
            ).transpose(),
        )

    def _check_inputs(self):
        assert self.fluor_vals.shape == self.conc_nm.shape
        # assert self.conc_nm[:,-1].sum() == 0

    def _check_survival_valid(self):
        if (self.fluor_dead >= self.fluor_untreated).any():
            self.status = "fail"
            self.note = "Highest concentration has more fluorescence than untreated."

    def _fit_model(
        self,
        conc_nm_log10: np.ndarray,
        percent_survival_fit: np.ndarray,
        bootstrap: bool = False,
    ) -> tuple[float, float, float, float]:
        """Fit the IC50"""

        # Make initial guess
        p0 = [
            percent_survival_fit.min(),
            percent_survival_fit.max(),
            np.median(conc_nm_log10),  # initial IC50 guess, we are in logspace
            1.0,  # initial hill slope guess
        ]

        # Set bounds
        bounds = [(-1_000, -1_000, -2, 0.01), (1_000, 1_000, conc_nm_log10.max(), 20)]

        # Attempt the fit
        try:
            fit_params, _ = curve_fit(
                f=hill_equation,
                xdata=conc_nm_log10.flatten(),
                ydata=percent_survival_fit.flatten(),
                p0=p0,
                bounds=bounds,
            )
            if not bootstrap:
                self.status = "pass" if self.status != "fail" else "fail"
            if np.isclose(fit_params[2], bounds[0][2]):
                self.status = "fail"
                self.note = "Estimated IC50 value has hit lower bound."
            if np.isclose(fit_params[2], bounds[1][2]):
                self.status = "fail"
                self.note = "Estimated IC50 value has hit upper bound."
            if fit_params[0] >= fit_params[1]:
                self.status = "fail"
                self.note = "Bottom parameter estimate higher than top."

        except RuntimeError:
            fit_params = (np.nan, np.nan, np.nan, np.nan)
            if not bootstrap:
                self.status = "fail"
                self.note = "Unable to fit the hill equation."

        return fit_params

    def _boostrap_ic50_ci(self, n_boot: int = 100):
        if not self.status == "pass":
            return np.nan, np.nan

        # We flatten for random sampling
        flat_ic50 = self.conc_nm_log10.flatten()
        flat_survive = self.percent_survival_fit.flatten()
        all_idx = np.arange(flat_ic50.shape[0])

        # Bootstrap
        boot_ic50s = []
        for _ in range(n_boot):
            idx = np.random.choice(all_idx, size=all_idx.shape[0], replace=True)
            boot_params = self._fit_model(
                flat_ic50[idx], flat_survive[idx], bootstrap=True
            )
            boot_ic50s.append(boot_params[2])

        # Get percentiles
        ci_low, ci_high = np.percentile(np.array(boot_ic50s), [2.5, 97.5])

        return ci_low, ci_high
    
    def _calc_r2(self, conc_nm_log10: np.ndarray, percent_survival_fit: np.ndarray,
                 fit_params: tuple[float, float, float, float]) -> float:
        """
        Calculate the R2 of the fit
        """
        pred = hill_equation(conc_nm_log10.flatten(), *fit_params)
        ss_res = np.sum((percent_survival_fit.flatten() - pred)**2)
        ss_tot = np.sum((percent_survival_fit.flatten() - percent_survival_fit.flatten().mean())**2)
        return 1 - ss_res/ss_tot

    def print_fit(self) -> None:
        print(f"  {self.drug.upper()} IC50 (nm): {10**self.fit_params[2]:.2f}")

    def plot_fit(self, output_path: str = None) -> None:
        """
        Plot the best fit

        """

        # Prepare the fit data
        _conc_nm_log10 = np.linspace(
            self.conc_nm_log10.min(), self.conc_nm_log10.max(), 200
        )
        _percent_survival = hill_equation(_conc_nm_log10, *self.fit_params)

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        # Fit
        ax.plot(10**_conc_nm_log10, _percent_survival, color="darkgrey", label="fit")

        # Values
        for i in range(self.n_reps):
            ax.scatter(
                x=10 ** self.conc_nm_log10[:, i],
                y=self.percent_survival_fit[:, i],
                label=f"rep {i}",
                ec="black",
                lw=0.5,
            )

        ax.set_xlim(10**-2, 10**4)

        if not np.isnan(self.fit_params[2]):
            ax.axvline(
                10 ** self.fit_params[2], color="forestgreen", label="IC50", ls="dashed"
            )

        ax.set_xscale("log")

        # Labels
        title = f"{self.sample_id}\n{self.drug.upper()} $-$ {self.status} $-$ IC50 {10**self.fit_params[2]:.1f}nm $-$ {self.r2:.1f}"
        ax.set_title(title, loc="left")
        ax.set_xlabel("nM [log10]")
        ax.set_ylabel("Survival (%)")
        ax.legend(bbox_to_anchor=(1, 1), loc="upper left")

        if output_path is not None:
            fig.savefig(
                f"{output_path}/ic50_fit.{self.sample_id}.{self.drug}.pdf",
                bbox_inches="tight",
                pad_inches=0.5,
                dpi=300,
            )
            plt.close()

    def get_fit(self) -> IC50Result:
        if self.bootstrap:
            return IC50ResultWithBootstrap(
                sample_id=self.sample_id,
                drug=self.drug,
                ic50=10 ** self.fit_params[2],
                r2=self.r2,
                ic50_lowci=10**self.ic50_lowci,
                ic50_highci=10**self.ic50_highci,
                status=self.status,
                note=self.note,
            )
        return IC50Result(
            sample_id=self.sample_id,
            drug=self.drug,
            ic50=10 ** self.fit_params[2],
            r2=self.r2,
            status=self.status,
            note=self.note,
        )


# --------------------------------------------------------------------------------
# VISUALS
#
# --------------------------------------------------------------------------------


def plot_plate_heatmap(
    df_plate: pd.DataFrame,
    values: str,
    vname: str = "Survival (%)",
    vmin: float = None,
    vmax: float = None,
    output_path: str = None,
) -> None:
    """
    Plot a plate heatmap
    """

    # Get row and column
    df_plate["row"] = [w[0] for w in df_plate["well"]]
    df_plate["col"] = [int(w[1:]) for w in df_plate["well"]]

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    cbar = ax.scatter(
        x="col",
        y="row",
        c=values,
        cmap="inferno",
        s=800,
        ec="black",
        vmin=vmin,
        vmax=vmax,
        data=df_plate,
    )

    ax.set_xticks(PLATE.cols, labels=PLATE.cols, fontweight="bold")
    ax.set_yticks(PLATE.rows, labels=PLATE.rows, fontweight="bold")

    ax.tick_params(
        top=False, left=False, labeltop=True, bottom=False, labelbottom=False
    )

    # Colorbar
    plt.colorbar(cbar, pad=0.05, label=vname)
    for s in ax.spines:
        ax.spines[s].set_visible(False)

    # axis
    ax.invert_yaxis()

    if output_path is not None:
        fig.savefig(
            f"{output_path}/platemap.{values}.pdf",
            bbox_inches="tight",
            pad_inches=0.5,
            dpi=300,
        )
        plt.close()


def plot_ic50s(df_ic50_pass: pd.DataFrame, output_path: str = None) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    sns.stripplot(
        x="drug",
        y="ic50",
        hue="drug",
        s=8,
        edgecolor="black",
        linewidth=0.5,
        data=df_ic50_pass,
        ax=ax,
    )

    ax.set_ylim(10**-2, 10**3)
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.5, color="darkgrey")
    ax.grid(axis="y", which="minor", alpha=0.5, linestyle="dotted", color="darkgrey")

    ax.set_xlabel("Drug")
    ax.set_ylabel("IC50 (nm)")

    if output_path is not None:
        fig.savefig(
            f"{output_path}/ic50_passing.pdf",
            bbox_inches="tight",
            pad_inches=0.5,
            dpi=300,
        )
        plt.close()


# --------------------------------------------------------------------------------
# Main
#
# --------------------------------------------------------------------------------


def main(dir_input: str, dir_output: str) -> None:
    """
    Process IC50 results
    """
    print("=" * 80)
    print("IC50 ANALYSIS")
    print("-" * 80)

    # Prepare inputs
    input_txts = glob.glob(f"{dir_input}/*.txt")
    print(f"Found {len(input_txts)} input text files.")

    # Get metadata
    df_metadata = pd.read_csv("metadata.csv", dtype={"sample1": str, "sample2": str, "sample3": str})
    df_metadata.set_index("plate", drop=True, inplace=True)
    print("Found metadata file:")
    print(tabulate(df_metadata, headers="keys"))

    # Prepare outputs
    os.makedirs(dir_output, exist_ok=True)

    results = []
    for input_txt in input_txts:
        print("-" * 80)
        print(f"Processing: {input_txt}")
        print("-" * 80)

        # Prepare inputs
        plate_name = os.path.basename(input_txt).split(".")[0]
        dir_plate = f"{dir_output}/{plate_name}"
        os.makedirs(dir_plate, exist_ok=True)
        try:
            plate_info = df_metadata.loc[plate_name]
            samples = [str(s) for s in plate_info[["sample1", "sample2", "sample3"]] if not pd.isna(s)]
            dilution_factor = plate_info["dilution_factor"].astype(int)
        except KeyError:
            raise KeyError(
                f"Could not find plate: {plate_name} in metadata file! Check spelling."
            )

        print("Loading raw data...")
        df_data = load_raw_data(input_txt)
        print(f" Samples: {', '.join(samples)}.")
        print(f" Dilution factor: {dilution_factor}")
        df_plate = get_plate_layout(samples=samples, dilution_factor=dilution_factor)

        print("Combining with plate layout...")
        df_data = pd.merge(
            left=df_plate,
            right=df_data[["well", "au"]],
            on="well",
            how="right",
            validate="1:1",
        )
        df_data.to_csv(f"{dir_plate}/combined_data.csv", index=False)

        print("Plotting raw fluorescence of plate...")
        plot_plate_heatmap(
            df_data, values="au", vname="AU (465-510)", output_path=dir_plate
        )

        print("Trying to determine IC50...")
        plate_results = []
        for (sample_id, drug), df_assay in df_data.groupby(["sample_id", "drug"]):
            print(f" Sample: {sample_id} | Drug: {drug.upper()}")
            print(" Fitting...")
            assay = IC50Assay.from_dataframe(df_assay)
            assay.print_fit()
            assay.plot_fit(output_path=dir_plate)
            plate_results.append(assay.get_fit())
        df_results = pd.DataFrame(plate_results)
        df_results.insert(0, "plate", plate_name)
        results.append(df_results)

    # Final IC50 table
    df_ic50 = pd.concat(results, axis=0, ignore_index=True)
    df_ic50.to_csv(f"{DIR_OUTPUT}/ic50.csv", index=False)

    print("")
    print("OVERVIEW")
    print(
        tabulate(
            pd.crosstab(
                df_ic50["drug"], df_ic50["status"], margins=True, margins_name="total"
            ),
            headers="keys",
        )
    )
    print("")

    plot_ic50s(df_ic50_pass=df_ic50.query("status == 'pass'"),
               output_path=dir_output)

    print("-" * 80)
    print("Done.")
    print("=" * 80)


if __name__ == "__main__":
    main(dir_input=DIR_INPUT, dir_output=DIR_OUTPUT)
