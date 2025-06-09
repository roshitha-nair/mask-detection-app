import splitfolders

input_folder = "original_dataset"  # your folder containing 'with_mask' and 'without_mask' folders
output_folder = "dataset"    # folder where train/test folders will be created

splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.8, .2)) 


print(" Dataset successfully split and copied.")
