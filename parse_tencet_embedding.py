import pickle

def write_file(path):
    content = [[0]*200, [1]*200]
    with open("./processed/embedding.pkl", 'wb+') as f1:
        count = 2
        with open(path, "r+", encoding='utf-8') as f:
            with open('./processed/word_to_index.pkl', 'wb+') as p:
                word_to_id = {'unk': 0, 'pad': 1}

                line = f.readline()

                while line:
                    line = f.readline()
                    line_split = line.split()
                    line_sp = line.split()[0]
                    if len(line_sp) == 1:

                        try:
                            assert(len(line_split)==201)
                            content.append([float(num)
                                            for num in line_split[1:]])
                            word_to_id[line_sp] = count
                            count += 1
                        except:
                            print(len(line_split))
                    if count == 1000:
                        print(line_sp)
                        print('write')
                        pickle.dump(content, f1)
                        pickle.dump(word_to_id, p)


                pickle.dump(word_to_id, p)
            print(count)


            pickle.dump(f1, content)

if __name__ == '__main__':
    path = "F:\我的下载\Tencent_AILab_ChineseEmbedding.txt"
    write_file(path)



