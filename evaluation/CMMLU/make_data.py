import csv
import json
import os
import random
random.seed(42)

def csv_to_list(filepath):
    """
    Convert a CSV file to a list of questions.
    """
    questions = []
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            questions.append({
                'question': row[1],
                'res1': row[2],
                'res2': row[3],
                'res3': row[4],
                'res4': row[5],
                'ans': row[6]
            })
    return questions[1:]

def generate_json(val_dir, test_dir, output_dir):

    
    # Process test directory
    for filename in os.listdir(test_dir):
        test_filepath = os.path.join(test_dir, filename)
        test_data = csv_to_list(test_filepath)
        val_filepath = os.path.join(val_dir, filename.replace('test', 'val'))
        demo_list = csv_to_list(val_filepath)
        demo_list = random.sample(demo_list, 5)
        print(val_filepath, len(demo_list))
        final_data = []
        for item in test_data:
            entry = {
                'demo': demo_list,
                'data': item
            }
            final_data.append(entry)
        
        # Save to JSON
        json_filename = filename.replace('.csv', '.json')
        json_filepath = os.path.join(output_dir, json_filename)
        with open(json_filepath, 'w', encoding='utf-8') as file:
            json.dump(final_data, file, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    val_dir = 'path_to_cmmlu_dev_folder'
    test_dir = 'path_to_cmmlu_test_folder'
    output_dir = './data/5shot'  # 替换为你的目标输出文件夹
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    generate_json(val_dir, test_dir, output_dir)
