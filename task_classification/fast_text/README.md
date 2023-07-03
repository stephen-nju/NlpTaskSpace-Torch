# 数据预处理
考虑具体的语料，可以采用不同的方式进行tokenizer.中文的话，常见的就可以使用jieba分词，或者采用单个字符，也可以使用Bertwordpiece等


# 数据格式

训练数据的训练格式label表示 __label__{标签名称}表示标签名称，标签名和数据之间用Tab键隔开,数据使用空格进行分词


示例：__label__shouji	索尼 手机 5iV

# 环境配置

python=3.7.4 
fasttext
