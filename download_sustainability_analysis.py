import wget
import zipfile
from constants import ROOT
import os


def main():

    ZIP_PATH = str(ROOT / "Sustainability_Analysis.zip")
    OUT_PATH = str(ROOT / "Sustainability_Analysis/")

    if not os.path.exists(ZIP_PATH):
        print("\ndownloading Sustainability_Analysis.zip...")
        wget.download(
            "https://zenodo.org/record/4564072/files/Sustainability_Analysis.zip?download=1",
            ZIP_PATH,
        )
    else:
        print("\nSustainability_Analysis.zip already exists")
    if not os.path.exists(OUT_PATH):
        print("\nextracting Sustainability_Analysis...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(OUT_PATH)
    else:
        print("\nextracted Sustainability_Analysis already exists")


if __name__ == "__main__":
    main()
