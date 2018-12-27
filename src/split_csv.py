import pandas as pd

def split_csv(data_file, base_dir):
    with open(data_file, 'r') as f:
        ignore = f.readline()
        object_id = None
        lines = []
        for line in f:
            fields = line.split(',')
            if object_id is None or fields[0] == object_id:
                lines.append(line)
                if object_id is None:
                    object_id = fields[0]
            else:
                with open('{}/{}.csv'.format(base_dir, object_id), 'w') as cs:
                    cs.writelines(lines)
                object_id = fields[0]
                lines = [line]
        if len(lines) > 0:
            with open('{}/{}.csv'.format(base_dir, object_id), 'w') as cs:
                    cs.writelines(lines)

def gen_objects_csv(meta_file, base_dir):
    df_meta = pd.read_csv(meta_file)
    objects = pd.object_id.drop_duplicates().values
    objects = [str(obj) for obj in objects]
    objects_str = ",".join(objects)
    
    with open("{}/objects.csv", "w") as f:
        f.write(objects_str)

gen_objects_csv("../input/training_set_metadata.csv", "../input/train/train_csv")
split_csv("../input/training_set.csv", "../input/train/train_csv")