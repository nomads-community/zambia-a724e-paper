import os
import sys
import glob
import shutil
import subprocess
import pandas as pd
from dataclasses import dataclass
from shlex import quote as q


# Parameters
REF_FASTA = "reference/PlasmoDB-67_Pfalciparum3D7_Genome.fasta.gz"
REF_GFF = "reference/PlasmoDB-67_Pfalciparum3D7.gff"
SAMPLES_CSV = "sample_set/table.zambia_sanger.all_info.csv"
ASSAYS = {
     "assay1": ("11109621661_Zambia_k13_Run2_SCF_SEQ_ABI", "K13_rev"),
     "assay2": ("11109805199_Zambia_K13_nested_Run2_SCF_SEQ_ABI", "p235_rev")
}

def create_vcf_from_ab1(
        ref_fasta: str,
        sample_ab1: str,
        output_prefix: str = None,
        peak_ratio: float = 0.25) -> str:
    """Create a *.vcf file from an *.ab1 file"""

    if not os.path.exists(sample_ab1):
        raise FileNotFoundError(f"No *.ab1 at {sample_ab1}.")
    
    if not sample_ab1.endswith(".ab1"):
        raise ValueError(f"Chromatogram file '{sample_ab1}' must end with *.ab1.")

    if output_prefix is None:
        output_prefix = sample_ab1.replace(".ab1", "")

    # Run the command
    cmd = ["tracy", "decompose", "-v", "-p", str(peak_ratio), "-r", ref_fasta, "-o", output_prefix, sample_ab1]
    subprocess.run(cmd, check=True)

    return f"{output_prefix}.bcf" # default path to BCF file

def annotate_consequences(
        input_vcf: str,
        output_vcf: str,
        ref_fasta: str,
        ref_gff: str) -> str:
    """Annotate the VCF using `bcftools csq`"""
    cmd = [
        "bcftools", "csq",
        "--local-csq",
        "-f", ref_fasta, 
        "-g", ref_gff, 
        "--phase", "a", 
        "-Oz", "-o", output_vcf, 
        input_vcf
    ]
    subprocess.run(cmd, check=True)

def convert_vcf_to_csv(input_vcf: str, output_csv: str, sample_id: str = None) -> str:
    """Convert a VCF to a CSV"""
    if sample_id is None:
            sample_id = os.path.basename(input_vcf).split(".")[0]
    cmd = (
        f"printf 'sample_id,chrom,pos,ref,alt,csq,gt\n' > {q(output_csv)} &&"
        f" bcftools query -f '{sample_id},%CHROM,%POS,%REF,%ALT,%BCSQ,[%GT]\n' {q(input_vcf)} >> {q(output_csv)}"
    )
    subprocess.run(cmd, shell=True, check=True)
    return output_csv

def convert_vcf_to_tsv(input_vcf: str, output_tsv: str, sample_id: str = None) -> str:
    """Convert a VCF to a TSV"""
    if sample_id is None:
        sample_id = os.path.basename(input_vcf).split(".")[0]
    cmd = (
        f"printf 'sample_id\\tchrom\\tpos\\tref\\talt\\tcsq\\tgt\\n' > {q(output_tsv)} && "
        f"bcftools query -f '{sample_id}\\t%CHROM\\t%POS\\t%REF\\t%ALT\\t%BCSQ\\t[%GT]\\n' "
        f"{q(input_vcf)} >> {q(output_tsv)}"
    )
    subprocess.run(cmd, shell=True, check=True)
    return output_tsv


def main(samples_csv: str, ab1_dir: str, primer: str, ref_fasta: str, ref_gff: str, results_dir: str="results"):
    """
    Compute VCF files
    """

    # Load metadata
    df_samples = pd.read_csv(samples_csv, dtype={"sample_id": str})

    # Process individually
    ix = 0
    for _, row in df_samples.iterrows():
        # Collect the *.ab1
        sample_id = str(row['sample_id'])
        source_ab1 = glob.glob(f"{results_dir}/0eurofins_raw/{ab1_dir}/{sample_id}_{primer}*.ab1")    
        print("-"*80)
        print(f"SAMPLE: {sample_id}")
        if not len(source_ab1) == 1:
            # I believe the skipped samples are the ones that could not be found.
            print(f"Found {len(source_ab1)} *.ab1 files for {sample_id}! Skipping.")
            continue
        
        # Copy and rename it
        sample_ab1 = f"{results_dir}/1ab1_files/{sample_id}.ab1"
        shutil.copy(src=source_ab1[0], dst=sample_ab1)
                     
        # Convert to BCF
        sample_prefix = f"{results_dir}/2tracy_outputs/{sample_id}"
        try:
            sample_bcf = create_vcf_from_ab1(
                ref_fasta=ref_fasta,
                sample_ab1=sample_ab1,
                output_prefix=sample_prefix
            )
        except subprocess.CalledProcessError:
             print(f"Failed for sample {sample_id}!")
             continue

        # Annotate
        sample_vcf = f"{results_dir}/3vcfs/{sample_id}.vcf"
        annotate_consequences(
            input_vcf=sample_bcf,
            output_vcf=sample_vcf,
            ref_fasta=ref_fasta,
            ref_gff=ref_gff
        )

        # To *.csv
        output_csv = f"{results_dir}/4tsvs/{sample_id}.tsv"
        convert_vcf_to_tsv(
            input_vcf=sample_vcf,
            output_tsv=output_csv,
            sample_id=sample_id
        )
        print("-"*80)
        ix += 1

    print(f"Succesfully processed {ix} samples.")


if __name__ == "__main__":

    assay = ASSAYS.get(sys.argv[1])
    if assay is None:
         raise ValueError(f"Can't find assay '{sys.argv[1]}'. Choose from: {', '.join(ASSAYS.keys())}.")

    main(samples_csv=SAMPLES_CSV,
         ab1_dir=assay[0],
         primer=assay[1],
         ref_fasta=REF_FASTA,
         ref_gff=REF_GFF,
         results_dir=f"results/{sys.argv[1]}")