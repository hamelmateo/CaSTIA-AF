# scripts/generate_merged_datasets.py

from calcium_activity_characterization.analysis.merger import merge_all_datasets
from pathlib import Path

image_sequences = [
    #20250404 datasets
    {
        "path": "D:/Mateo/20250326/Output/IS1",
        "date": "20250326",
        "image_sequence": "IS1",
        "experiment_type": "spontaneous",
        "condition": "control - 1st run",
        "confluency": 1200000,
        "concentration": None,
        "time": None
    },
    {
        "path": "D:/Mateo/20250326/Output/IS2",
        "date": "20250326",
        "image_sequence": "IS2",
        "experiment_type": "spontaneous",
        "condition": "control - 2nd run",
        "confluency": 1200000,
        "concentration": None,
        "time": None
    },
    {
        "path": "D:/Mateo/20250326/Output/IS3",
        "date": "20250326",
        "image_sequence": "IS3",
        "experiment_type": "spontaneous",
        "condition": "control - 1st run",
        "confluency": 1200000,
        "concentration": None,
        "time": None
    },
    {
        "path": "D:/Mateo/20250326/Output/IS4",
        "date": "20250326",
        "image_sequence": "IS4",
        "experiment_type": "spontaneous",
        "condition": "control - 2nd run",
        "confluency": 1200000,
        "concentration": None,
        "time": None
    },
    #20250404 datasets
    {
        "path": "D:/Mateo/20250404/Output/IS1",
        "date": "20250404",
        "image_sequence": "IS1",
        "experiment_type": "spontaneous",
        "condition": "control - 1st run",
        "confluency": 1200000,
        "concentration": None,
        "time": None
    },
    {
        "path": "D:/Mateo/20250404/Output/IS2",
        "date": "20250404",
        "image_sequence": "IS2",
        "experiment_type": "spontaneous",
        "condition": "control - 2nd run",
        "confluency": 1200000,
        "concentration": None,
        "time": None
    },
    {
        "path": "D:/Mateo/20250404/Output/IS3",
        "date": "20250404",
        "image_sequence": "IS3",
        "experiment_type": "spontaneous",
        "condition": "control - 1st run",
        "confluency": 1200000,
        "concentration": None,
        "time": None
    },
    {
        "path": "D:/Mateo/20250404/Output/IS4",
        "date": "20250404",
        "image_sequence": "IS4",
        "experiment_type": "spontaneous",
        "condition": "unuseful",
        "confluency": 1200000,
        "concentration": None,
        "time": None
    },
    {
        "path": "D:/Mateo/20250404/Output/IS5",
        "date": "20250404",
        "image_sequence": "IS5",
        "experiment_type": "spontaneous",
        "condition": "unuseful",
        "confluency": 1200000,
        "concentration": None,
        "time": None
    },
    #20250409 datasets
    {
        "path": "D:/Mateo/20250409/Output/IS01",
        "date": "20250409",
        "image_sequence": "IS01",
        "experiment_type": "spontaneous",
        "condition": "control - 1st run",
        "confluency": 600000,
        "concentration": None,
        "time": None
    },
    {
        "path": "D:/Mateo/20250409/Output/IS02",
        "date": "20250409",
        "image_sequence": "IS02",
        "experiment_type": "spontaneous",
        "condition": "control - 2nd run",
        "confluency": 600000,
        "concentration": None,
        "time": None
    },
    {
        "path": "D:/Mateo/20250409/Output/IS03",
        "date": "20250409",
        "image_sequence": "IS03",
        "experiment_type": "spontaneous",
        "condition": "control - 1st run",
        "confluency": 600000,
        "concentration": None,
        "time": None
    },
    {
        "path": "D:/Mateo/20250409/Output/IS04",
        "date": "20250409",
        "image_sequence": "IS04",
        "experiment_type": "spontaneous",
        "condition": "control - 2nd run",
        "confluency": 600000,
        "concentration": None,
        "time": None
    },
    {
        "path": "D:/Mateo/20250409/Output/IS05",
        "date": "20250409",
        "image_sequence": "IS05",
        "experiment_type": "spontaneous",
        "condition": "control - 1st run",
        "confluency": 120000,
        "concentration": None,
        "time": None
    },
    {
        "path": "D:/Mateo/20250409/Output/IS06",
        "date": "20250409",
        "image_sequence": "IS06",
        "experiment_type": "spontaneous",
        "condition": "control - 2nd run",
        "confluency": 120000,
        "concentration": None,
        "time": None
    },
    {
        "path": "D:/Mateo/20250409/Output/IS07",
        "date": "20250409",
        "image_sequence": "IS07",
        "experiment_type": "spontaneous",
        "condition": "control - 1st run",
        "confluency": 120000,
        "concentration": None,
        "time": None
    },
    {
        "path": "D:/Mateo/20250409/Output/IS08",
        "date": "20250409",
        "image_sequence": "IS08",
        "experiment_type": "spontaneous",
        "condition": "control - 2nd run",
        "confluency": 120000,
        "concentration": None,
        "time": None
    },
    {
        "path": "D:/Mateo/20250409/Output/IS09",
        "date": "20250409",
        "image_sequence": "IS09",
        "experiment_type": "spontaneous",
        "condition": "control - 1st run",
        "confluency": 1200000,
        "concentration": None,
        "time": None
    },
    {
        "path": "D:/Mateo/20250409/Output/IS10",
        "date": "20250409",
        "image_sequence": "IS10",
        "experiment_type": "spontaneous",
        "condition": "control - 2nd run",
        "confluency": 1200000,
        "concentration": None,
        "time": None
    },
    {
        "path": "D:/Mateo/20250409/Output/IS11",
        "date": "20250409",
        "image_sequence": "IS11",
        "experiment_type": "spontaneous",
        "condition": "control - 1st run",
        "confluency": 1200000,
        "concentration": None,
        "time": None
    },
    {
        "path": "D:/Mateo/20250409/Output/IS12",
        "date": "20250409",
        "image_sequence": "IS12",
        "experiment_type": "spontaneous",
        "condition": "control - 2nd run",
        "confluency": 1200000,
        "concentration": None,
        "time": None
    },
    {
        "path": "D:/Mateo/20250409/Output/IS13",
        "date": "20250409",
        "image_sequence": "IS13",
        "experiment_type": "spontaneous",
        "condition": "unuseful",
        "confluency": 1200000,
        "concentration": None,
        "time": None
    },
    #20250416 datasets
    {
        "path": "D:/Mateo/20250416/Output/IS1",
        "date": "20250416",
        "image_sequence": "IS1",
        "experiment_type": "spontaneous",
        "condition": "control - 1st run",
        "confluency": 600000,
        "concentration": None,
        "time": None
    },
    {
        "path": "D:/Mateo/20250416/Output/IS2",
        "date": "20250416",
        "image_sequence": "IS2",
        "experiment_type": "spontaneous",
        "condition": "control - 2nd run",
        "confluency": 600000,
        "concentration": None,
        "time": None
    },
    {
        "path": "D:/Mateo/20250416/Output/IS3",
        "date": "20250416",
        "image_sequence": "IS3",
        "experiment_type": "spontaneous",
        "condition": "control - 1st run",
        "confluency": 600000,
        "concentration": None,
        "time": None
    },
    {
        "path": "D:/Mateo/20250416/Output/IS4",
        "date": "20250416",
        "image_sequence": "IS4",
        "experiment_type": "spontaneous",
        "condition": "control - 2nd run",
        "confluency": 600000,
        "concentration": None,
        "time": None
    },
    {
        "path": "D:/Mateo/20250416/Output/IS5",
        "date": "20250416",
        "image_sequence": "IS5",
        "experiment_type": "spontaneous",
        "condition": "unuseful",
        "confluency": 600000,
        "concentration": None,
        "time": None
    },
    {
        "path": "D:/Mateo/20250416/Output/IS6",
        "date": "20250416",
        "image_sequence": "IS6",
        "experiment_type": "spontaneous",
        "condition": "unuseful",
        "confluency": 600000,
        "concentration": None,
        "time": None
    },
    #20250424 datasets - stimulated with ACH
    {
        "path": "D:/Mateo/20250424/Output/IS1",
        "date": "20250424",
        "image_sequence": "IS1",
        "experiment_type": "stimulated",
        "condition": "ACH - 1st run",
        "confluency": 1200000,
        "concentration": "100nM",
        "time": "-300s"
    },
    {
        "path": "D:/Mateo/20250424/Output/IS2",
        "date": "20250424",
        "image_sequence": "IS2",
        "experiment_type": "stimulated",
        "condition": "ACH - 2nd run",
        "confluency": 1200000,
        "concentration": "100nM",
        "time": "+1500s"
    },
    {
        "path": "D:/Mateo/20250424/Output/IS3",
        "date": "20250424",
        "image_sequence": "IS3",
        "experiment_type": "stimulated",
        "condition": "ACH - 1st run",
        "confluency": 1200000,
        "concentration": "100nM",
        "time": "-300s"
    },
    {
        "path": "D:/Mateo/20250424/Output/IS4",
        "date": "20250424",
        "image_sequence": "IS4",
        "experiment_type": "stimulated",
        "condition": "ACH - 2nd run",
        "confluency": 1200000,
        "concentration": "100nM",
        "time": "+1500s"
    },
    {
        "path": "D:/Mateo/20250424/Output/IS5",
        "date": "20250424",
        "image_sequence": "IS5",
        "experiment_type": "stimulated",
        "condition": "ACH - 1st run",
        "confluency": 1200000,
        "concentration": "100nM",
        "time": "+2d"
    },
    {
        "path": "D:/Mateo/20250424/Output/IS6",
        "date": "20250424",
        "image_sequence": "IS6",
        "experiment_type": "stimulated",
        "condition": "ACH - 2nd run",
        "confluency": 1200000,
        "concentration": "100nM",
        "time": "+2d"
    },
    {
        "path": "D:/Mateo/20250424/Output/IS7",
        "date": "20250424",
        "image_sequence": "IS7",
        "experiment_type": "stimulated",
        "condition": "ACH - 1st run",
        "confluency": 1200000,
        "concentration": "100nM",
        "time": "+2d"
    },
    {
        "path": "D:/Mateo/20250424/Output/IS8",
        "date": "20250424",
        "image_sequence": "IS8",
        "experiment_type": "stimulated",
        "condition": "ACH - 2nd run",
        "confluency": 1200000,
        "concentration": "100nM",
        "time": "+2d"
    },
    {
        "path": "D:/Mateo/20250424/Output/IS9",
        "date": "20250424",
        "image_sequence": "IS9",
        "experiment_type": "stimulated",
        "condition": "unuseful",
        "confluency": 1200000,
        "concentration": "100nM",
        "time": "+2d"
    },
    #20250501 datasets - stimulated with ACH
    {
        "path": "D:/Mateo/20250501/Output/IS01",
        "date": "20250501",
        "image_sequence": "IS01",
        "experiment_type": "stimulated",
        "condition": "Unuseful",
        "confluency": 300000,
        "concentration": "10uM",
        "time": "-300s"
    },
    {
        "path": "D:/Mateo/20250501/Output/IS02",
        "date": "20250501",
        "image_sequence": "IS02",
        "experiment_type": "stimulated",
        "condition": "ACH - 2nd run",
        "confluency": 300000,
        "concentration": "10uM",
        "time": "+1500s"
    },
    {
        "path": "D:/Mateo/20250501/Output/IS03",
        "date": "20250501",
        "image_sequence": "IS03",
        "experiment_type": "stimulated",
        "condition": "ACH - 1st run",
        "confluency": 300000,
        "concentration": "10uM",
        "time": "-300s"
    },
    {
        "path": "D:/Mateo/20250501/Output/IS04",
        "date": "20250501",
        "image_sequence": "IS04",
        "experiment_type": "stimulated",
        "condition": "ACH - 2nd run",
        "confluency": 300000,
        "concentration": "10uM",
        "time": "+1500s"
    },
    {
        "path": "D:/Mateo/20250501/Output/IS05",
        "date": "20250501",
        "image_sequence": "IS05",
        "experiment_type": "stimulated",
        "condition": "Unuseful",
        "confluency": 300000,
        "concentration": "10uM",
        "time": "+3300s"
    },
    {
        "path": "D:/Mateo/20250501/Output/IS06",
        "date": "20250501",
        "image_sequence": "IS06",
        "experiment_type": "stimulated",
        "condition": "ACH - 1st run",
        "confluency": 300000,
        "concentration": "10uM",
        "time": "+2d"
    },
    {
        "path": "D:/Mateo/20250501/Output/IS07",
        "date": "20250501",
        "image_sequence": "IS07",
        "experiment_type": "stimulated",
        "condition": "ACH - 2nd run",
        "confluency": 300000,
        "concentration": "10uM",
        "time": "+2d"
    },
    {
        "path": "D:/Mateo/20250501/Output/IS08",
        "date": "20250501",
        "image_sequence": "IS08",
        "experiment_type": "stimulated",
        "condition": "ACH - 1st run",
        "confluency": 300000,
        "concentration": "10uM",
        "time": "+2d"
    },
    {
        "path": "D:/Mateo/20250501/Output/IS09",
        "date": "20250501",
        "image_sequence": "IS09",
        "experiment_type": "stimulated",
        "condition": "ACH - 2nd run",
        "confluency": 300000,
        "concentration": "10uM",
        "time": "+2d"
    },
    {
        "path": "D:/Mateo/20250501/Output/IS10",
        "date": "20250501",
        "image_sequence": "IS10",
        "experiment_type": "stimulated",
        "condition": "unuseful",
        "confluency": 300000,
        "concentration": "10uM",
        "time": "+2d"
    },
    #20250618 datasets - stimulated with ACH
    {
        "path": "D:/Mateo/20250618/Output/IS1",
        "date": "20250618",
        "image_sequence": "IS1",
        "experiment_type": "stimulated",
        "condition": "ACH - 1st run",
        "confluency": 1000000,
        "concentration": "10uM",
        "time": "+2d"
    },
    {
        "path": "D:/Mateo/20250618/Output/IS2",
        "date": "20250618",
        "image_sequence": "IS2",
        "experiment_type": "stimulated",
        "condition": "ACH - 1st run",
        "confluency": 1000000,
        "concentration": "10uM",
        "time": "+2d"
    },
    {
        "path": "D:/Mateo/20250618/Output/IS3",
        "date": "20250618",
        "image_sequence": "IS3",
        "experiment_type": "stimulated",
        "condition": "ACH - 1st run",
        "confluency": 1000000,
        "concentration": "1uM",
        "time": "+2d"
    },
    {
        "path": "D:/Mateo/20250618/Output/IS4",
        "date": "20250618",
        "image_sequence": "IS4",
        "experiment_type": "stimulated",
        "condition": "ACH - 1st run",
        "confluency": 1000000,
        "concentration": "1uM",
        "time": "+2d"
    },
    {
        "path": "D:/Mateo/20250618/Output/IS5",
        "date": "20250618",
        "image_sequence": "IS5",
        "experiment_type": "stimulated",
        "condition": "ACH - 1st run",
        "confluency": 1000000,
        "concentration": "100nM",
        "time": "+2d"
    },
    {
        "path": "D:/Mateo/20250618/Output/IS6",
        "date": "20250618",
        "image_sequence": "IS6",
        "experiment_type": "stimulated",
        "condition": "ACH - 1st run",
        "confluency": 1000000,
        "concentration": "100nM",
        "time": "+2d"
    },
    {
        "path": "D:/Mateo/20250618/Output/IS7",
        "date": "20250618",
        "image_sequence": "IS7",
        "experiment_type": "stimulated",
        "condition": "ACH - 1st run",
        "confluency": 1000000,
        "concentration": "10nM",
        "time": "+2d"
    },
    {
        "path": "D:/Mateo/20250618/Output/IS8",
        "date": "20250618",
        "image_sequence": "IS8",
        "experiment_type": "stimulated",
        "condition": "ACH - 1st run",
        "confluency": 1000000,
        "concentration": "10nM",
        "time": "+2d"
    },
    #20250624 datasets - stimulated with ACH
    {
        "path": "D:/Mateo/20250624/Output/IS01",
        "date": "20250624",
        "image_sequence": "IS01",
        "experiment_type": "spontaneous",
        "condition": "control - 1st run",
        "confluency": 1000000,
        "concentration": None,
        "time": None
    },
    {
        "path": "D:/Mateo/20250624/Output/IS02",
        "date": "20250624",
        "image_sequence": "IS02",
        "experiment_type": "stimulated",
        "condition": "unuseful",
        "confluency": 1000000,
        "concentration": "100nM",
        "time": "-50s"
    },
    {
        "path": "D:/Mateo/20250624/Output/IS03",
        "date": "20250624",
        "image_sequence": "IS03",
        "experiment_type": "spontaneous",
        "condition": "control - 1st run",
        "confluency": 1000000,
        "concentration": None,
        "time": None
    },
    {
        "path": "D:/Mateo/20250624/Output/IS04",
        "date": "20250624",
        "image_sequence": "IS04",
        "experiment_type": "stimulated",
        "condition": "unuseful",
        "confluency": 1000000,
        "concentration": "100nM",
        "time": "-50s"
    },
    {
        "path": "D:/Mateo/20250624/Output/IS05",
        "date": "20250624",
        "image_sequence": "IS05",
        "experiment_type": "stimulated",
        "condition": "ACH - 1st run",
        "confluency": 1000000,
        "concentration": "10uM",
        "time": "+2d"
    },
    {
        "path": "D:/Mateo/20250624/Output/IS06",
        "date": "20250624",
        "image_sequence": "IS06",
        "experiment_type": "stimulated",
        "condition": "ACH - 1st run",
        "confluency": 1000000,
        "concentration": "10uM",
        "time": "+2d"
    },
    {
        "path": "D:/Mateo/20250624/Output/IS07",
        "date": "20250624",
        "image_sequence": "IS07",
        "experiment_type": "stimulated",
        "condition": "ACH - 1st run",
        "confluency": 1000000,
        "concentration": "1uM",
        "time": "+2d"
    },
    {
        "path": "D:/Mateo/20250624/Output/IS08",
        "date": "20250624",
        "image_sequence": "IS08",
        "experiment_type": "stimulated",
        "condition": "ACH - 1st run",
        "confluency": 1000000,
        "concentration": "1uM",
        "time": "+2d"
    },
    {
        "path": "D:/Mateo/20250624/Output/IS09",
        "date": "20250624",
        "image_sequence": "IS09",
        "experiment_type": "stimulated",
        "condition": "ACH - 1st run",
        "confluency": 1000000,
        "concentration": "100nM",
        "time": "+2d"
    },
    {
        "path": "D:/Mateo/20250624/Output/IS10",
        "date": "20250624",
        "image_sequence": "IS10",
        "experiment_type": "stimulated",
        "condition": "ACH - 1st run",
        "confluency": 1000000,
        "concentration": "100nM",
        "time": "+2d"
    },
    #20250701 datasets - stimulated with ACH
    {
        "path": "D:/Mateo/20250701/Output/IS1",
        "date": "20250701",
        "image_sequence": "IS1",
        "experiment_type": "spontaneous",
        "condition": "control - 1st run",
        "confluency": 1200000,
        "concentration": None,
        "time": None
    },
    {
        "path": "D:/Mateo/20250701/Output/IS2",
        "date": "20250701",
        "image_sequence": "IS2",
        "experiment_type": "stimulated",
        "condition": "ACH - 1st run",
        "confluency": 1200000,
        "concentration": "10uM",
        "time": "-300s"
    },
    {
        "path": "D:/Mateo/20250701/Output/IS3",
        "date": "20250701",
        "image_sequence": "IS3",
        "experiment_type": "stimulated",
        "condition": "ACH - 1st run",
        "confluency": 1200000,
        "concentration": "10uM",
        "time": "-300s"
    },
    {
        "path": "D:/Mateo/20250701/Output/IS4",
        "date": "20250701",
        "image_sequence": "IS4",
        "experiment_type": "stimulated",
        "condition": "unuseful",
        "confluency": 1200000,
        "concentration": "10uM",
        "time": "+1500s"
    },
    {
        "path": "D:/Mateo/20250701/Output/IS5",
        "date": "20250701",
        "image_sequence": "IS5",
        "experiment_type": "stimulated",
        "condition": "ACH - 1st run",
        "confluency": 1200000,
        "concentration": "10uM",
        "time": "+1800s"
    },
    {
        "path": "D:/Mateo/20250701/Output/IS5",
        "date": "20250701",
        "image_sequence": "IS5",
        "experiment_type": "stimulated",
        "condition": "ACH - 1st run",
        "confluency": 1200000,
        "concentration": "10uM",
        "time": "+1800s"
    },
]

def main():
    output_dir = Path("D:/Mateo/Results")
    merge_all_datasets(image_sequences, output_dir)

if __name__ == "__main__":
    main()
