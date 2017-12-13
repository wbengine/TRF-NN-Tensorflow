单词模型数据，用来验证方法。

LDC_gigaword_en.words   原始数据文件，单词提取自LDC gigaword english数据集

models_ppl.txt          列出了KN模型的ppl和似然值

运行 perpare_data.py 可以自动分成如下文件：
train.words     训练集
valid.words     验证集
test.words      测试集
nbest.words     nbest list
transcript.words    nbest的正确结果