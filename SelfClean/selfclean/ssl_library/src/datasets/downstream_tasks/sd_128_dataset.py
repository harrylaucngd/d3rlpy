import re
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd

from ....src.datasets.generic_image_dataset import GenericImageDataset


class SD128Label(Enum):
    DISEASE = "diagnosis"
    MALIGNANT = "malignant"


class SD128Dataset(GenericImageDataset):
    """SD-128 dataset."""

    IMG_COL = "img_path"
    LBL_COL = "diagnosis"

    def __init__(
        self,
        dataset_dir: Union[str, Path] = "data/dataset/",
        transform=None,
        val_transform=None,
        label_col: SD128Label = SD128Label.DISEASE,
        return_path: bool = False,
        image_extensions: Sequence = ("*.png", "*.jpg", "*.JPEG"),
        data_quality_issues_list: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        """
        Initializes the dataset.

        Sets the correct path for the needed arguments.

        Parameters
        ----------
        dataset_dir : str
            Directory with all the images.
        transform : Union[callable, optional]
            Optional transform to be applied to the images.
        val_transform : Union[callable, optional]
            Optional transform to be applied to the images when in validation mode.
        return_path : bool
            If the path of the image should be returned or not.
        """
        super().__init__(
            dataset_dir=dataset_dir,
            transform=transform,
            val_transform=val_transform,
            return_path=return_path,
            image_extensions=image_extensions,
            **kwargs,
        )
        # post-process metadata
        meta_data = self.meta_data
        meta_data["diagnosis"] = meta_data["diagnosis"].apply(
            lambda x: re.sub(r"\(.*\)", "", x).strip()
        )
        meta_data["lbl_diagnosis"] = pd.factorize(meta_data["diagnosis"])[0]
        meta_data.reset_index(drop=True, inplace=True)
        self.meta_data = meta_data

        # categorize into benign/malignant
        benign_conditions = [
            "Follicular_Mucinosis",
            "Fixed_Drug_Eruption",
            "Steroid_Acne",
            "Nail_Dystrophy",
            "Onycholysis",
            "Infantile_Atopic_Dermatitis",
            "Angular_Cheilitis",
            "Actinic_solar_Damage",
            "Nevus_Spilus",
            "Sebaceous_Gland_Hyperplasia",
            "Compound_Nevus",
            "Syringoma",
            "Tinea_Pedis",
            "Scar",
            "Paronychia",
            "Keratoacanthoma",
            "Onychoschizia",
            "Granulation_Tissue",
            "Perioral_Dermatitis",
            "Livedo_Reticularis",
            "Lichen_Planus",
            "Blue_Nevus",
            "Tinea_Corporis",
            "Nummular_Eczema",
            "Nevus_Sebaceous_of_Jadassohn",
            "Pityriasis_Rosea",
            "Crowe's_Sign",
            "Stasis_Ulcer",
            "Dry_Skin_Eczema",
            "Striae",
            "Benign_Keratosis",
            "Androgenetic_Alopecia",
            "Median_Nail_Dystrophy",
            "Keloid",
            "Stasis_Dermatitis",
            "Seborrheic_Keratosis",
            "Congenital_Nevus",
            "Stasis_Edema",
            "Disseminated_Actinic_Porokeratosis",
            "Milia",
            "Leukonychia",
            "Allergic_Contact_Dermatitis",
            "Scalp_Psoriasis",
            "Epidermal_Nevus",
            "Eczema",
            "Hyperkeratosis_Palmaris_Et_Plantaris",
            "Myxoid_Cyst",
            "Tinea_Manus",
            "Desquamation",
            "Cutaneous_Horn",
            "Dermatofibroma",
            "Nail_Psoriasis",
            "Neurodermatitis",
            "Candidiasis",
            "Acne_Keloidalis_Nuchae",
            "Tinea_Faciale",
            "Dilated_Pore_of_Winer",
            "Herpes_Zoster",
            "Callus",
            "Lymphocytic_Infiltrate_of_Jessner",
            "Inverse_Psoriasis",
            "Exfoliative_Erythroderma",
            "Lipoma",
            "Lichen_Sclerosis_Et_Atrophicus",
            "Junction_Nevus",
            "Rosacea",
            "Pitted_Keratolysis",
            "Dysplastic_Nevus",
            "Pityrosporum_Folliculitis",
            "Radiodermatitis",
            "Dermatosis_Papulosa_Nigra",
            "Apocrine_Hydrocystoma",
            "Bowen's_Disease",
            "Cellulitis",
            "Drug_Eruption",
            "Steroid_Use_abusemisuse_Dermatitis",
            "Hypertrichosis",
            "Koilonychia",
            "Skin_Tag",
            "Keratosis_Pilaris",
            "Nevus_Incipiens",
            "Xerosis",
            "Rhinophyma",
            "Seborrheic_Dermatitis",
            "Verruca_Vulgaris",
            "Ichthyosis",
            "Herpes_Simplex_Virus",
            "Tinea_Cruris",
            "Dyshidrosiform_Eczema",
            "Pyogenic_Granuloma",
            "Pseudofolliculitis_Barbae",
            "Fibroma_Molle",
            "Tinea_Versicolor",
            "Kerion",
            "Angioma",
            "Pseudorhinophyma",
            "Neurofibroma",
            "Halo_Nevus",
            "Lichen_Simplex_Chronicus",
            "Keratolysis_Exfoliativa_of_Wende",
            "Beau's_Lines",
            "Erythema_Multiforme",
            "Epidermoid_Cyst",
            "Factitial_Dermatitis",
            "Alopecia_Areata",
            "Granuloma_Annulare",
            "Psoriasis",
            "Favre_Racouchot",
            "Onychomycosis",
            "Pustular_Psoriasis",
            "Erythema_Craquele",
            "Digital_Fibroma",
            "Clubbing_of_Fingers",
            "Guttate_Psoriasis",
            "Acne_Vulgaris",
            "Hidradenitis_Suppurativa",
            "Ulcer",
            "Leukocytoclastic_Vasculitis",
        ]

        malignant_conditions = [
            "Malignant_Melanoma",
            "Lentigo_Maligna_Melanoma",
            "Bowen's_Disease",
            "Metastatic_Carcinoma",
            "Basal_Cell_Carcinoma",
            "Keratoacanthoma",
            "Darier-White_Disease",
        ]

        self.meta_data["malignant"] = self.meta_data["diagnosis"].apply(
            lambda x: (
                False
                if x in benign_conditions
                else True if x in malignant_conditions else None
            )
        )
        self.meta_data["lbl_malignant"] = self.meta_data["malignant"].astype(int)

        # remove data quality issues if file is given
        self.remove_data_quality_issues(data_quality_issues_list)
        self.meta_data.reset_index(drop=True, inplace=True)

        # Global configs
        self.LBL_COL = f"lbl_{label_col.value}"
        self.return_path = return_path
        self.classes = (
            self.meta_data["diagnosis"].unique().tolist()
            if label_col == SD128Label.DISEASE
            else ["benign", "malignant"]
        )
        self.n_classes = len(self.classes)
