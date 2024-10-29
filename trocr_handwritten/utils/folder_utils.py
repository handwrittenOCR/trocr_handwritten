import os
import subprocess
import logging


#   25/10/2024: new code created that does :
#   Checks if there are subfolders in the main path.
#   If subfolders exist, it processes each subfolder individually, creating corresponding subfolders in the output paths.
#   If no subfolders exist, it processes the main folder directly.
#   Allows for additional arguments to be passed to the script via the additional_args parameter.


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_folders(main_path, output_paths, script_path, script_args, additional_args=None):
    """
    Process folders or files in the main path, maintaining the structure in output paths.
    Works for both cases: with and without subfolders.
    
    :param main_path: Path to the main input folder (e.g., PATH_PAGES)
    :param output_paths: List of paths to output folders (e.g., [PATH_XML, PATH_LINES])
    :param script_path: Path to the script to be executed
    :param script_args: List of argument names for output paths
    :param additional_args: Dictionary of additional arguments for the script
    """
    def process_path(input_path, output_paths):
        # Construct the command
        command = f"python {script_path} --input_path {input_path} "
        for arg, path in zip(script_args, output_paths):
            command += f"--{arg} {path} "
        
        if additional_args:
            for key, value in additional_args.items():
                command += f"--{key} {value} "
        
        # Execute the command
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        
        # Print the output
        for line in process.stdout:
            print(line.decode('utf-8', errors='ignore').strip())
        
        # Wait for the process to finish and get the return code
        process.wait()
        return process.returncode

    # Check if main_path contains subfolders
    subfolders = [f for f in os.listdir(main_path) if os.path.isdir(os.path.join(main_path, f))]
    
    if subfolders:
        # Process each subfolder
        for subfolder in subfolders:
            logging.info(f"Processing subfolder: {subfolder}")
            subfolder_path = os.path.join(main_path, subfolder)
            
            # Create corresponding subfolders in output paths
            subfolder_output_paths = []
            for output_path in output_paths:
                subfolder_output_path = os.path.join(output_path, subfolder)
                os.makedirs(subfolder_output_path, exist_ok=True)
                subfolder_output_paths.append(subfolder_output_path)
            
            return_code = process_path(subfolder_path, subfolder_output_paths)
            
            if return_code == 0:
                logging.info(f"Subfolder {subfolder} processed successfully")
            else:
                logging.error(f"Processing of subfolder {subfolder} failed with return code {return_code}")
    else:
        # Process the main folder directly
        logging.info(f"Processing main folder: {main_path}")
        return_code = process_path(main_path, output_paths)
        
        if return_code == 0:
            logging.info(f"Main folder processed successfully")
        else:
            logging.error(f"Processing of main folder failed with return code {return_code}")

    logging.info("All processing completed")