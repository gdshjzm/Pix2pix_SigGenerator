import os

filename = os.listdir('./generated_image/')
number = list(range(1,153))
filename_list = [str(i) + '.png' for i in number]
name_map = dict(zip(filename,filename_list))

def rename_files_in_folders(folder_a, name_mapping):
    # 遍历A文件夹中的文件
    for filename in os.listdir(folder_a):
        if filename in name_mapping:
            old_file_path = os.path.join(folder_a, filename)
            new_file_path = os.path.join(folder_a, name_mapping[filename])
            os.rename(old_file_path, new_file_path)
            print(f'Renamed: {old_file_path} -> {new_file_path}')
            
input_img_path = './generated_image/'
target_img_path = './targeted_image/'
         
rename_files_in_folders(input_img_path,name_map)
rename_files_in_folders(target_img_path,name_map)