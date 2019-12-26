#_*_coding:utf-8_*_
# author    : jmx
# create    : 19-12-24 下午3:46
# filename  : seq2seq.py
# IDE   : PyCharm

import numpy as np
import tensorflow as tf
from pathlib import Path
import pickle
from tensorflow import layers

from tensorflow.python.ops import array_ops
from tensorflow.contrib import seq2seq
from tensorflow.contrib.seq2seq import BahdanauAttention
from tensorflow.contrib.seq2seq import LuongAttention
from tensorflow.contrib.seq2seq import AttentionWrapper
from tensorflow.contrib.seq2seq import BeamSearchDecoder

from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.rnn import MultiRNNCell  #实现多层RNN
from tensorflow.contrib.rnn import DropoutWrapper  #drop网络
from tensorflow.contrib.rnn import ResidualWrapper #残差网络  就是把输入concat到输出上一起返回


train_txt = '/home/ytkj/lz/1224Seq2Seq/data/train/train_train.txt'
all_txt = '/home/ytkj/lz/1224Seq2Seq/data/map_file/all.txt'
all_txt2 = '/home/ytkj/lz/1224Seq2Seq/data/map_file/all_.txt'

use_bidir = True
use_attention = True
# # English source data
# with open(all_txt, "r", encoding="utf-8") as f:
# 	source_text = f.readlines()
#
# # French target data
# with open(all_txt2, "r", encoding="utf-8") as f:
# 	target_text = f.readlines()
#
# source_vocab = set()
# target_vocab = set()
#
# for line in source_text:
# 	for char in line:
# 		source_vocab.add(char)
#
# for line in target_text:
# 	for char in line:
# 		target_vocab.add(char)
# source_vocab = list(source_vocab)
# target_vocab = list(target_vocab)
#
#
# # 特殊字符
# SOURCE_CODES = ['<PAD>', '<UNK>']
# TARGET_CODES = ['<PAD>', '<EOS>', '<UNK>', '<GO>']  # 在target中，需要增加<GO>与<EOS>特殊字符
#
# # 构造英文映射字典
# source_vocab_to_int = {word: idx for idx, word in enumerate(SOURCE_CODES + source_vocab)}
# source_int_to_vocab = {idx: word for idx, word in enumerate(SOURCE_CODES + source_vocab)}
# source_vocab_to_int_file = open('source_vocab_to_int.pickle', 'wb')
# pickle.dump(source_vocab_to_int, source_vocab_to_int_file)
# source_vocab_to_int_file.close()
# source_int_to_vocab_file = open('source_int_to_vocab.pickle', 'wb')
# pickle.dump(source_int_to_vocab, source_int_to_vocab_file)
# source_int_to_vocab_file.close()
#
# # 构造法语映射词典
# target_vocab_to_int = {word: idx for idx, word in enumerate(TARGET_CODES + target_vocab)}
# target_int_to_vocab = {idx: word for idx, word in enumerate(TARGET_CODES + target_vocab)}
# target_vocab_to_int_file = open('target_vocab_to_int.pickle', 'wb')
# pickle.dump(target_vocab_to_int, target_vocab_to_int_file)
# target_vocab_to_int_file.close()
# target_int_to_vocab_file = open('target_int_to_vocab.pickle', 'wb')
# pickle.dump(target_int_to_vocab, target_int_to_vocab_file)
# target_int_to_vocab_file.close()

source_vocab_to_int_file = open('source_vocab_to_int.pickle', 'rb')
source_vocab_to_int = pickle.load(source_vocab_to_int_file)
source_int_to_vocab_file = open('source_int_to_vocab.pickle', 'rb')
source_int_to_vocab = pickle.load(source_int_to_vocab_file)

target_vocab_to_int_file = open('target_vocab_to_int.pickle', 'rb')
target_vocab_to_int = pickle.load(target_vocab_to_int_file)
target_int_to_vocab_file = open('target_int_to_vocab.pickle', 'rb')
target_int_to_vocab = pickle.load(target_int_to_vocab_file)

def text_to_int(sentence, map_dict, max_length=20, is_target=False):
	"""
	对文本句子进行数字编码

	@param sentence: 一个完整的句子，str类型
	@param map_dict: 单词到数字的映射，dict
	@param max_length: 句子的最大长度
	@param is_target: 是否为目标语句。在这里要区分目标句子与源句子，因为对于目标句子（即翻译后的句子）我们需要在句子最后增加<EOS>
	"""

	# 用<PAD>填充整个序列
	text_to_idx = []
	# unk index
	unk_idx = map_dict.get("<UNK>")
	pad_idx = map_dict.get("<PAD>")
	eos_idx = map_dict.get("<EOS>")

	# 如果是输入源文本
	if not is_target:
		for word in sentence:
			text_to_idx.append(map_dict.get(word, unk_idx))

	# 否则，对于输出目标文本需要做<EOS>的填充最后
	else:
		for word in sentence:
			text_to_idx.append(map_dict.get(word, unk_idx))
		text_to_idx.append(eos_idx)

	# 如果超长需要截断
	if len(text_to_idx) > max_length:
		return text_to_idx[:max_length]
	# 如果不够则增加<PAD>
	else:
		text_to_idx = text_to_idx + [pad_idx] * (max_length - len(text_to_idx))
		return text_to_idx

source_train_lines = open(train_txt, encoding='utf-8').readlines()
target_train_lines = open(train_txt.replace('_train.txt', '.txt'), encoding='utf-8').readlines()

source_text_to_int = []
for line in source_train_lines:
	line = line.strip()
	source_text_to_int.append(text_to_int(line, source_vocab_to_int, 51, is_target=False)) #51
target_text_to_int = []
for line in target_train_lines:
	line = line.strip()
	target_text_to_int.append(text_to_int(line, target_vocab_to_int, 83, # 83
										  is_target=True))

X = np.array(source_text_to_int)
Y = np.array(target_text_to_int)
indexs = np.arange(len(source_text_to_int))
source_text_to_int = np.array(source_text_to_int)
target_text_to_int = np.array(target_text_to_int)
np.random.shuffle(indexs)
source_text_to_int = list(source_text_to_int[indexs])
target_text_to_int = list(target_text_to_int[indexs])
print(indexs)

def model_inputs():
	"""
	构造输入

	返回：inputs, targets, learning_rate, source_sequence_len, target_sequence_len, max_target_sequence_len，类型为tensor
	"""
	inputs = tf.placeholder(tf.int32, [None, None], name="inputs")
	targets = tf.placeholder(tf.int32, [None, None], name="targets")
	learning_rate = tf.placeholder(tf.float32, name="learning_rate")

	source_sequence_len = tf.placeholder(tf.int32, (None,), name="source_sequence_len")
	target_sequence_len = tf.placeholder(tf.int32, (None,), name="target_sequence_len")
	max_target_sequence_len = tf.placeholder(tf.int32, (None,), name="max_target_sequence_len")

	return inputs, targets, learning_rate, source_sequence_len, target_sequence_len, max_target_sequence_len


def encoder_layer(rnn_inputs, rnn_size, rnn_num_layers,
				  source_sequence_len, source_vocab_size, encoder_embedding_size=100):
	"""
	构造Encoder端

	@param rnn_inputs: rnn的输入
	@param rnn_size: rnn的隐层结点数
	@param rnn_num_layers: rnn的堆叠层数
	@param source_sequence_len: 英文句子序列的长度
	@param source_vocab_size: 英文词典的大小
	@param encoder_embedding_size: Encoder层中对单词进行词向量嵌入后的维度
	"""
	# 对输入的单词进行词向量嵌入
	encoder_embed = tf.contrib.layers.embed_sequence(rnn_inputs, source_vocab_size, encoder_embedding_size)

	# LSTM单元
	def get_lstm(rnn_size):
		lstm = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123))
		return lstm

	# 堆叠rnn_num_layers层LSTM
	lstms = tf.contrib.rnn.MultiRNNCell([get_lstm(rnn_size) for _ in range(rnn_num_layers)])
	if not use_bidir:
		encoder_outputs, encoder_states = tf.nn.dynamic_rnn(lstms, encoder_embed, source_sequence_len,
														dtype=tf.float32)
	else:
		encoder_cell_bw = tf.contrib.rnn.MultiRNNCell([get_lstm(rnn_size) for _ in range(rnn_num_layers)])
		(
			(encoder_fw_outputs, encoder_bw_outputs),
			(encoder_fw_state, encoder_bw_state)
		) = tf.nn.bidirectional_dynamic_rnn(  # 动态多层双向lstm_rnn
			cell_fw=lstms,
			cell_bw=encoder_cell_bw,
			inputs=encoder_embed,
			sequence_length=source_sequence_len,
			dtype=tf.float32,
			swap_memory=True
		)
		encoder_outputs = tf.concat([encoder_fw_outputs, encoder_bw_outputs], 2)

		encoder_state = []
		for i in range(rnn_num_layers):
			encoder_state.append(encoder_fw_state[i])
			encoder_state.append(encoder_bw_state[i])
		encoder_states = tuple(encoder_state)
	return encoder_outputs, encoder_states


def decoder_layer_inputs(target_data, target_vocab_to_int, batch_size):
	"""
	对Decoder端的输入进行处理

	@param target_data: 法语数据的tensor
	@param target_vocab_to_int: 法语数据的词典到索引的映射
	@param batch_size: batch size
	"""
	# 去掉batch中每个序列句子的最后一个单词
	ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
	# 在batch中每个序列句子的前面添加”<GO>"
	decoder_inputs = tf.concat([tf.fill([batch_size, 1], target_vocab_to_int["<GO>"]),
								ending], 1)

	return decoder_inputs


def decoder_layer_train(encoder_states, decoder_cell, decoder_embed,
						target_sequence_len, max_target_sequence_len, output_layer):
	"""
	Decoder端的训练

	@param encoder_states: Encoder端编码得到的Context Vector
	@param decoder_cell: Decoder端
	@param decoder_embed: Decoder端词向量嵌入后的输入
	@param target_sequence_len: 法语文本的长度
	@param max_target_sequence_len: 法语文本的最大长度
	@param output_layer: 输出层
	"""

	# 生成helper对象
	training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed,
														sequence_length=target_sequence_len,
														time_major=False)

	training_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
													   training_helper,
													   encoder_states,
													   output_layer)

	training_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
																	   impute_finished=True,
																	   maximum_iterations=max_target_sequence_len)

	return training_decoder_outputs


def decoder_layer_infer(encoder_states, decoder_cell, decoder_embed, start_id, end_id,
						max_target_sequence_len, output_layer, batch_size):
	"""
	Decoder端的预测/推断

	@param encoder_states: Encoder端编码得到的Context Vector
	@param decoder_cell: Decoder端
	@param decoder_embed: Decoder端词向量嵌入后的输入
	@param start_id: 句子起始单词的token id， 即"<GO>"的编码
	@param end_id: 句子结束的token id，即"<EOS>"的编码
	@param max_target_sequence_len: 法语文本的最大长度
	@param output_layer: 输出层
	@batch_size: batch size
	"""

	start_tokens = tf.tile(tf.constant([start_id], dtype=tf.int32), [batch_size], name="start_tokens")

	inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embed,
																start_tokens,
																end_id)

	inference_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
														inference_helper,
														encoder_states,
														output_layer)

	inference_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
																		impute_finished=True,
																		maximum_iterations=max_target_sequence_len)

	return inference_decoder_outputs


def cell_input_fn(inputs, attention):
	""" 根据attn_input_feeding属性来判断是否在attention计算前进行一次投影的计算"""

	attn_projection = layers.Dense(256,
	                               dtype=tf.float32,
	                               use_bias=False,
	                               name='attention_cell_input_fn')

	return attn_projection(array_ops.concat([inputs, attention], -1))

def build_decoder_cell(encoder_outputs, encoder_states, rnn_size, rnn_num_layers,
                       batch_size, source_sequence_len):
	def get_lstm(rnn_size):
		lstm = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=456))
		return lstm

	if use_bidir:
		encoder_states = encoder_states[-rnn_num_layers:]

	cell = tf.contrib.rnn.MultiRNNCell([get_lstm(rnn_size) for _ in range(rnn_num_layers)])

	if use_attention:
		attention_mechanism = BahdanauAttention(
			num_units=256,
			memory=encoder_outputs,
			memory_sequence_length=source_sequence_len
		)


		attention_cell = AttentionWrapper(
			cell=cell,
			attention_mechanism=attention_mechanism,
			attention_layer_size=256,
			cell_input_fn=cell_input_fn,
			name='AttentionWrapper'
		)
		# 空状态
		decoder_initial_state = attention_cell.zero_state(batch_size, tf.float32)

		# 传递encoder的状态  定义decoder阶段的初始化状态，直接使用encoder阶段的最后一个隐层状态进行赋值
		decoder_initial_state = decoder_initial_state.clone(
			cell_state=encoder_states
		)
		return attention_cell, decoder_initial_state

	return cell, encoder_states


def decoder_layer(encoder_states, decoder_inputs, target_sequence_len,
				  max_target_sequence_len, rnn_size, rnn_num_layers,
				  target_vocab_to_int, target_vocab_size, decoder_embedding_size, batch_size, cell):
	"""
	构造Decoder端

	@param encoder_states: Encoder端编码得到的Context Vector
	@param decoder_inputs: Decoder端的输入
	@param target_sequence_len: 法语文本的长度
	@param max_target_sequence_len: 法语文本的最大长度
	@param rnn_size: rnn隐层结点数
	@param rnn_num_layers: rnn堆叠层数
	@param target_vocab_to_int: 法语单词到token id的映射
	@param target_vocab_size: 法语词典的大小
	@param decoder_embedding_size: Decoder端词向量嵌入的大小
	@param batch_size: batch size
	"""

	decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoder_embedding_size]))
	decoder_embed = tf.nn.embedding_lookup(decoder_embeddings, decoder_inputs)

	def get_lstm(rnn_size):
		lstm = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=456))
		return lstm

	# decoder_cell = tf.contrib.rnn.MultiRNNCell([get_lstm(rnn_size) for _ in range(rnn_num_layers)])
	decoder_cell = cell
	# output_layer logits
	output_layer = tf.layers.Dense(target_vocab_size)

	with tf.variable_scope("decoder"):
		training_logits = decoder_layer_train(encoder_states,
											  decoder_cell,
											  decoder_embed,
											  target_sequence_len,
											  max_target_sequence_len,
											  output_layer)

	with tf.variable_scope("decoder", reuse=True):
		inference_logits = decoder_layer_infer(encoder_states,
											   decoder_cell,
											   decoder_embeddings,
											   target_vocab_to_int["<GO>"],
											   target_vocab_to_int["<EOS>"],
											   max_target_sequence_len,
											   output_layer,
											   batch_size)

	return training_logits, inference_logits


def seq2seq_model(input_data, target_data, batch_size,
				  source_sequence_len, target_sequence_len, max_target_sentence_len,
				  source_vocab_size, target_vocab_size,
				  encoder_embedding_size, decoder_embeding_size,
				  rnn_size, rnn_num_layers, target_vocab_to_int):
	"""
	构造Seq2Seq模型

	@param input_data: tensor of input data
	@param target_data: tensor of target data
	@param batch_size: batch size
	@param source_sequence_len: 英文语料的长度
	@param target_sequence_len: 法语语料的长度
	@param max_target_sentence_len: 法语的最大句子长度
	@param source_vocab_size: 英文词典的大小
	@param target_vocab_size: 法语词典的大小
	@param encoder_embedding_size: Encoder端词嵌入向量大小
	@param decoder_embedding_size: Decoder端词嵌入向量大小
	@param rnn_size: rnn隐层结点数
	@param rnn_num_layers: rnn堆叠层数
	@param target_vocab_to_int: 法语单词到token id的映射
	"""
	encoder_outputs, encoder_states = encoder_layer(input_data, rnn_size, rnn_num_layers, source_sequence_len,
									  source_vocab_size, encoder_embedding_size)

	decoder_inputs = decoder_layer_inputs(target_data, target_vocab_to_int, batch_size)

	cell, encoder_states = build_decoder_cell(encoder_outputs=encoder_outputs, encoder_states=encoder_states,
	                                          rnn_size=rnn_size, rnn_num_layers=rnn_num_layers,
	                                          batch_size=batch_size, source_sequence_len=source_sequence_len)

	training_decoder_outputs, inference_decoder_outputs = decoder_layer(encoder_states,
																		decoder_inputs,
																		target_sequence_len,
																		max_target_sentence_len,
																		rnn_size,
																		rnn_num_layers,
																		target_vocab_to_int,
																		target_vocab_size,
																		decoder_embeding_size,
																		batch_size,
	                                                                    cell=cell)
	return training_decoder_outputs, inference_decoder_outputs


def get_batches(sources, targets, batch_size):
	"""
	获取batch
	"""

	for batch_i in range(0, len(sources) // batch_size):
		start_i = batch_i * batch_size

		# Slice the right amount for the batch
		sources_batch = sources[start_i:start_i + batch_size]
		targets_batch = targets[start_i:start_i + batch_size]

		# Need the lengths for the _lengths parameters
		targets_lengths = []
		for target in targets_batch:
			targets_lengths.append(len(target))

		source_lengths = []
		for source in sources_batch:
			source_lengths.append(len(source))

		yield sources_batch, targets_batch, source_lengths, targets_lengths

def main():
	# Number of Epochs
	epochs = 100
	# Batch Size
	batch_size = 128 # 128
	# RNN Size
	rnn_size = 128
	# Number of Layers
	rnn_num_layers = 1
	# Embedding Size
	encoder_embedding_size = 100
	decoder_embedding_size = 100
	# Learning Rate
	lr = 0.001
	# 每50轮打一次结果
	display_step = 50
	restore = True


	if not restore:
		train_graph = tf.Graph()

		with train_graph.as_default():
			inputs, targets, learning_rate, source_sequence_len, target_sequence_len, _ = model_inputs()
			max_target_sequence_len = 83
			train_logits, inference_logits = seq2seq_model(tf.reverse(inputs, [-1]),
														   targets,
														   batch_size,
														   source_sequence_len,
														   target_sequence_len,
														   max_target_sequence_len,
														   len(source_vocab_to_int),
														   len(target_vocab_to_int),
														   encoder_embedding_size,
														   decoder_embedding_size,
														   rnn_size,
														   rnn_num_layers,
														   target_vocab_to_int)

			training_logits = tf.identity(train_logits.rnn_output, name="logits")
			inference_logits = tf.identity(inference_logits.sample_id, name="predictions")
			masks = tf.sequence_mask(target_sequence_len, max_target_sequence_len, dtype=tf.float32, name="masks")

			with tf.name_scope("optimization"):
				cost = tf.contrib.seq2seq.sequence_loss(training_logits, targets, masks)

				optimizer = tf.train.AdamOptimizer(learning_rate)

				gradients = optimizer.compute_gradients(cost)
				clipped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
				train_op = optimizer.apply_gradients(clipped_gradients)

		with tf.Session(graph=train_graph) as sess:
			sess.run(tf.global_variables_initializer())
			saver = tf.train.Saver()
			for epoch_i in range(epochs):
				for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
						get_batches(source_text_to_int, target_text_to_int, batch_size)):

					_, loss = sess.run(
						[train_op, cost],
						{inputs: source_batch,
						 targets: target_batch,
						 learning_rate: lr,
						 source_sequence_len: sources_lengths,
						 target_sequence_len: targets_lengths})

					if batch_i % display_step == 0 and batch_i > 0:
						batch_train_logits = sess.run(
							inference_logits,
							{inputs: source_batch,
							 source_sequence_len: sources_lengths,
							 target_sequence_len: targets_lengths})
						source_sentences = []
						for sentence in source_batch:
							temp = ''
							for i in sentence:
								temp += source_int_to_vocab[i]
							source_sentences.append(temp.replace('<PAD>', '').replace('<EOS>', ''))
						trans_sentences = []
						for sentence in batch_train_logits:
							temp = ''
							for i in sentence:
								temp += target_int_to_vocab[i]
							trans_sentences.append(temp.replace('<PAD>', '').replace('<EOS>', ''))
						print('source:{}, \ntrain predict {}'.format(source_sentences, trans_sentences))
						print('Epoch {:>3} Batch {:>4}/{} - Loss: {:>6.4f}'
							  .format(epoch_i, batch_i, len(source_text_to_int) // batch_size, loss))

				# Save Model

				saver.save(sess, "checkpoints_attention/dev",global_step=epoch_i)
				print('Model Trained and Saved')
	else:
		loaded_graph = tf.Graph()
		with tf.Session(graph=loaded_graph) as sess:
			# Load saved model
			loader = tf.train.import_meta_graph('checkpoints_attention/dev-89.meta')
			loader.restore(sess, tf.train.latest_checkpoint('./checkpoints_attention'))
			p = Path(tf.train.latest_checkpoint('./checkpoints_attention'))
			step = int(p.stem.split('-')[1])
			print(step)
			inputs = loaded_graph.get_tensor_by_name('inputs:0')
			targets = loaded_graph.get_tensor_by_name('targets:0')
			learning_rate = loaded_graph.get_tensor_by_name('learning_rate:0')
			logits = loaded_graph.get_tensor_by_name('predictions:0')
			target_sequence_len = loaded_graph.get_tensor_by_name('target_sequence_len:0')
			source_sequence_len = loaded_graph.get_tensor_by_name('source_sequence_len:0')
			train_op = loaded_graph.get_operation_by_name('optimization/Adam')
			cost = loaded_graph.get_tensor_by_name('optimization/sequence_loss/truediv:0')

			saver = tf.train.Saver()
			for epoch_i in range(epochs):
				for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
						get_batches(source_text_to_int, target_text_to_int, batch_size)):

					_, loss = sess.run(
						[train_op, cost],
						{inputs: source_batch,
						 targets: target_batch,
						 learning_rate: lr,
						 source_sequence_len: sources_lengths,
						 target_sequence_len: targets_lengths})

					if batch_i % display_step == 0 and batch_i > 0:
						batch_train_logits = sess.run(
							logits,
							{inputs: source_batch,
							 source_sequence_len: sources_lengths,
							 target_sequence_len: targets_lengths})
						source_sentences = []
						for sentence in source_batch:
							temp = ''
							for i in sentence:
								temp += source_int_to_vocab[i]
							source_sentences.append(temp.replace('<PAD>', '').replace('<EOS>', ''))
						trans_sentences = []
						for sentence in batch_train_logits:
							temp = ''
							for i in sentence:
								temp += target_int_to_vocab[i]
							trans_sentences.append(temp.replace('<PAD>', '').replace('<EOS>', ''))
						print('source:{}, \ntrain predict {}'.format(source_sentences, trans_sentences))
						print('Epoch {:>3} Batch {:>4}/{} - Loss: {:>6.4f}'
						      .format(epoch_i, batch_i, len(source_text_to_int) // batch_size, loss))

				# Save Model

				saver.save(sess, "checkpoints_attention/dev", global_step=epoch_i + step + 1)
				print('Model Trained and Saved')



def sentence_to_seq(sentence, source_vocab_to_int):
	"""
	将句子转化为数字编码
	"""
	unk_idx = source_vocab_to_int["<UNK>"]
	word_idx = [source_vocab_to_int.get(word, unk_idx) for word in sentence]

	return word_idx

def sentence_to_seq_pad(sentence, source_vocab_to_int, maxLen):
	"""
	将句子转化为数字编码
	"""
	unk_idx = source_vocab_to_int["<UNK>"]
	pad_idx = source_vocab_to_int['<PAD>']
	word_idx = [source_vocab_to_int.get(word, unk_idx) for word in sentence]
	if len(word_idx) < maxLen:
		word_idx = word_idx + [pad_idx] * (maxLen - len(word_idx))

	if len(word_idx) > maxLen:
		word_idx = word_idx[:maxLen]

	return word_idx

def predict():
	batch_size = 128
	translate_sentence_text = input("请输入句子：")
	translate_sentence = sentence_to_seq_pad(translate_sentence_text, source_vocab_to_int, 51)

	loaded_graph = tf.Graph()
	with tf.Session(graph=loaded_graph) as sess:
		# Load saved model
		loader = tf.train.import_meta_graph('checkpoints_attention/dev-49.meta')
		loader.restore(sess, tf.train.latest_checkpoint('./checkpoints_attention'))

		input_data = loaded_graph.get_tensor_by_name('inputs:0')
		logits = loaded_graph.get_tensor_by_name('predictions:0')
		target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_len:0')
		source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_len:0')

		a = [translate_sentence] * batch_size
		b = [len(translate_sentence) * 2] * batch_size
		c = [len(translate_sentence)] * batch_size

		translate_logits = sess.run(logits, {input_data: [translate_sentence] * batch_size,
											 # target_sequence_length: [len(translate_sentence) * 2] * batch_size,
											 # source_sequence_length: [len(translate_sentence)] * batch_size})[0]
											 target_sequence_length: [len(translate_sentence) * 2] * batch_size,
											 source_sequence_length: [len(translate_sentence)] * batch_size})
		translate_logits = translate_logits[0]

	print('【Input】')
	print('  Word Ids:      {}'.format([i for i in translate_sentence]))
	print('  English Words: {}'.format([source_int_to_vocab[i] for i in translate_sentence]))

	print('\n【Prediction】')
	print('  Word Ids:      {}'.format([i for i in translate_logits]))
	print('  French Words: {}'.format([target_int_to_vocab[i] for i in translate_logits]))

	print("\n【Full Sentence】")
	print(" ".join([target_int_to_vocab[i] for i in translate_logits]))

def predict_file(filename):
	batch_size = 128
	lines = open(filename, encoding='utf-8').readlines()
	lines_r = open(filename.replace('_train.txt','.txt'), encoding='utf-8').readlines()
	train_sentences = []
	train_lens = []
	results = []
	results_lens = []
	result_ids = []
	all_lens = len(lines)
	TP = 0
	for line in lines:
		train_sentences.append(line.strip())
		# train_lens.append(len(line.strip()))
	for line in lines_r:
		results.append(line.strip())
		results_lens.append(len(line.strip()))
		result_ids.append(sentence_to_seq_pad(line, target_vocab_to_int, 83))
	train_ids = []
	for line in train_sentences:
		train_ids.append(sentence_to_seq_pad(line, source_vocab_to_int, 51))
		train_lens.append(51)
	# translate_sentence_text = input("请输入句子：")
	# translate_sentence = sentence_to_seq(translate_sentence_text, source_vocab_to_int)
	trans_sentences = []
	loaded_graph = tf.Graph()
	with tf.Session(graph=loaded_graph) as sess:
		# Load saved model
		loader = tf.train.import_meta_graph('checkpoints_attention/dev-88.meta')
		loader.restore(sess, tf.train.latest_checkpoint('./checkpoints_attention'))

		input_data = loaded_graph.get_tensor_by_name('inputs:0')
		logits = loaded_graph.get_tensor_by_name('predictions:0')
		target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_len:0')
		source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_len:0')

		for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
				get_batches(train_ids, result_ids, batch_size)):
			translate_logits = sess.run(logits, {input_data: source_batch,
											 # target_sequence_length: [len(translate_sentence) * 2] * batch_size,
											 # source_sequence_length: [len(translate_sentence)] * batch_size})[0]
											 target_sequence_length: targets_lengths,
											 source_sequence_length: sources_lengths})
			for i in translate_logits:
				trans_sentences.append(i)
	p = Path(filename)
	fw = open(p.stem + '_attention_results.txt', encoding='utf-8', mode='w')
	for i, source_id in enumerate(trans_sentences):
		trans_id = trans_sentences[i]
		source_sentence = ''
		trans_sentence = ''
		result = results[i]
		# for id in source_id:
		# 	source_sentence += source_int_to_vocab[id]
		for id in trans_id:
			trans_sentence += target_int_to_vocab[id]
		trans_sentence = trans_sentence.replace('<PAD>', '').replace('<EOS>', '')
		if trans_sentence == result:
			TP += 1
		acc = TP / all_lens
		fw.write(trans_sentence + '\t' + result + '\n')
	fw.write('预测句子级别准确率:' + str(acc))
	fw.close()
predict_file('/home/ytkj/lz/1224Seq2Seq/data/valid/valid_train.txt')
predict_file('/home/ytkj/lz/1224Seq2Seq/data/train/train_train.txt')
# predict()
# main()