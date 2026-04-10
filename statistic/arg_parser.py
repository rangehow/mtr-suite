import argparse

def parse_args():
 """parseparameter"""
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", help="Model name or path.")
    parser.add_argument("--processed_dataset_path",)
    parser.add_argument("--sharegpt_path",)
    parser.add_argument("--domain_classifier_path",)
    parser.add_argument("--domain_result_path",)
    parser.add_argument("--coral_document_dir",)
    parser.add_argument("--coral_dataset_dir",)
    parser.add_argument("--mtr_test_path",)
    parser.add_argument("--mtr_train_path",)
    return parser.parse_args()