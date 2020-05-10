import time, datetime
import torch
import json
from torch import distributed as dist
from torch.autograd import Variable

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ModuleWithDependencies():
	def __init__(self, module_with_dependencies, rank, is_first=False, is_last=False, 
		prev_rank=None, next_rank=None):
		self._module = module_with_dependencies
		self._rank = rank
		self._module.cuda()
		# set完rank之后需要立刻拷贝到相应GPU上去
		self._is_first = is_first
		self._is_last = is_last
		self._prev_rank = prev_rank
		self._next_rank = next_rank
		self.input_tensors = None
		self.output_tensors = None


	def module(self):
		return self._module

	def prev_rank(self):
		return self._prev_rank

	def set_prev_rank(self, rank_id):
		self._prev_rank = rank_id

	def next_rank(self):
		return self._next_rank

	def set_next_rank(self, rank_id):
		self._next_rank = rank_id

	def is_first(self):
		return self._is_first

	def is_last(self):
		return self._is_last
	
	def cuda(self, id=None):
		if id == None:
			id = self._rank
		#self._module.to('cuda:' + str(id))
		self._module.cuda()
		self._rank = id

	def rank(self):
		return self._rank


class StageRuntime:

	def __init__(self, model, config):
		# Metadata needed for forward and backward pass within this stage.
		self._module_with_dependencies = []
		json_config_file = json.load(open(config, 'r'))
		self.module_to_stage_map = json_config_file.get("module_to_stage_map", None)
		self.stage_to_rank_map = json_config_file.get("stage_to_rank_map", None)
		self._parameters = []
		self._loss
		prev_module = None
		prev_prev_module = None
		index = 0

		assert len(self.module_to_stage_map) == len(model)

		# set modules except the last one
		for (stage, inputs, outputs) in model:
			if index == 0:
				prev_module = stage
				index += 1
				continue
			if index == 1:
				self._module_with_dependencies.append(
					ModuleWithDependencies(prev_module, 
						self.stage_to_rank_map[self.module_to_stage_map[index - 1]],
						is_first=True, next_rank=self.stage_to_rank_map[self.module_to_stage_map[index]]))
			else:
				self._module_with_dependencies.append(
					ModuleWithDependencies(prev_module,
						self.stage_to_rank_map[self.module_to_stage_map[index - 1]],
						prev_rank=self.stage_to_rank_map[self.module_to_stage_map[index - 2]],
						next_rank=self.stage_to_rank_map[self.module_to_stage_map[index]]))
			prev_prev_module = prev_module
			prev_module = stage
			index += 1

		# set last module
		self._module_with_dependencies.append(
			ModuleWithDependencies(prev_module,
				self.stage_to_rank_map[self.module_to_stage_map[index - 1]],
				is_last=True, prev_rank=self.stage_to_rank_map[self.module_to_stage_map[index - 2]]))

		# set parameters
		for i in range(len(self._module_with_dependencies)):
			self._parameters.extend(model[i][0].parameters())

	def scale(self, config):
		json_config_file = json.load(open(config, 'r'))
		self.module_to_stage_map = json_config_file.get("module_to_stage_map", None)
		self.stage_to_rank_map = json_config_file.get("stage_to_rank_map", None)
		
		for i in range(len(self._module_with_dependencies)):
			rank = self.stage_to_rank_map[self.module_to_stage_map[i]]
			if rank != self._module_with_dependencies[i].rank():
				self._module_with_dependencies[i].cuda(rank)
				if i != len(self._module_with_dependencies) - 1:
					self._module_with_dependencies[i + 1].set_prev_rank(rank)
				if i != 0:
					self._module_with_dependencies[i - 1].set_next_rank(rank)

	def module_with_dependencies(self):
		return self._module_with_dependencies

	def parameters(self):
		return self._parameters

	def run_forward(self, inputs, labels, one_hot_labels=False, num_classes=10):
		#input_tensors = inputs.to('cuda:' + str(self.module_with_dependencies()[0].rank()))
		input_tensors = inputs.cuda()
		if not one_hot_labels:
			# make labels one_hot
			labels = torch.sparse.torch.eye(num_classes).index_select(0, labels)
		#labels = labels.to('cuda:' + str(self.module_with_dependencies()[-1].rank()))
		labels = labels.cuda()
		output_tensors = None
		count = 0
		for eachmodule in self.module_with_dependencies():
			count += 1
			if eachmodule.is_first():
				eachmodule.input_tensors = input_tensors
				output_tensors = eachmodule.module()(eachmodule.input_tensors)
				eachmodule.output_tensors = output_tensors
			elif eachmodule.is_last():
				input_tensors = output_tensors
				eachmodule.input_tensors = input_tensors

				if eachmodule.rank() != eachmodule.prev_rank():
					"""
					output_tensors = eachmodule.module()(
						eachmodule.input_tensors.to('cuda:' + str(eachmodule.rank())), labels)
					"""
					output_tensors = eachmodule.module()(
						eachmodule.input_tensors.cuda(), labels)
				else:
					output_tensors = eachmodule.module()(eachmodule.input_tensors, labels)
				eachmodule.output_tensors = output_tensors

			else:
				input_tensors = output_tensors
				eachmodule.input_tensors = input_tensors
				if eachmodule.rank() != eachmodule.prev_rank():
					"""
					output_tensors = eachmodule.module()(
						*[eachtensor.to('cuda:' + str(eachmodule.rank())) for eachtensor in eachmodule.input_tensors])
					"""
					output_tensors = eachmodule.module()(
						*[eachtensor.to('cuda:' + str(eachmodule.rank())) for eachtensor in eachmodule.input_tensors])
				else:
					output_tensors = eachmodule.module()(*[eachtensor for eachtensor in eachmodule.input_tensors])
				eachmodule.output_tensors = output_tensors
		return output_tensors

	def run_backward(self, loss):
		output_tensors = loss
		input_tensors = None
		grad_tensors = None

		torch.autograd.backward(loss)

class ModelParallelRuntime:

	def __init__(self, model, rank, world_size, config, receive_parameters=False):
		# Metadata needed for forward and backward pass within this stage.
		self.model = model
		self.rank = int(rank)
		self._module_with_dependencies = []
		self.config = config
		json_config_file = json.load(open(config, 'r'))
		self.module_to_stage_map = json_config_file.get("module_to_stage_map", None)
		self.stage_to_rank_map = json_config_file.get("stage_to_rank_map", None)
		self._parameters = []
		self._loss = None
		self._losses = AverageMeter()
		self.process_groups = []
		self.optimizer = None
		prev_module = None
		prev_prev_module = None
		local_index = 0
		global_index = 0

		assert len(self.module_to_stage_map) == len(self.model)

		# make process_groups
		for i in range(world_size):
			sub_group = []
			for j in range(world_size):
				if i == j:
					sub_group.append(None)
				elif i > j:
					sub_group.append(None)
				else:
					sub_group.append(
						torch.distributed.new_group(ranks=[i, j], backend='nccl'))
			self.process_groups.append(sub_group)

		# set modules
		
		for (stage, inputs, outputs) in model:
			if self.stage_to_rank_map[self.module_to_stage_map[global_index]] != self.rank:
				global_index += 1
				continue

			if global_index == 0:
				prev_rank = None
				is_first = True
			else:
				prev_rank = self.stage_to_rank_map[self.module_to_stage_map[global_index - 1]]
				is_first = False
			if global_index == len(self.module_to_stage_map) - 1:
				is_last = True
				next_rank = None
			else:
				is_last = False
				next_rank = self.stage_to_rank_map[self.module_to_stage_map[global_index + 1]]

			self._module_with_dependencies.append(ModuleWithDependencies(stage(), 
				self.rank, prev_rank=prev_rank, next_rank=next_rank,
				is_first=is_first, is_last=is_last))
			
			local_index += 1
			global_index += 1


		# set parameters
		for i in range(len(self._module_with_dependencies)):
			self._parameters.extend(self._module_with_dependencies[i].module().parameters())
		if len(self._parameters) > 0:
			self.optimizer = torch.optim.SGD(self._parameters, lr=0.001)
		#print("rank: ", self.rank)
		#for each in self._module_with_dependencies:
		#	print(each.prev_rank(), each.rank(), each.next_rank(), each.is_first(), each.is_last())
	def module_with_dependencies(self):
		return self._module_with_dependencies

	def parameters(self):
		return self._parameters

	def loss(self):
		return self._losses.avg

	def reset_loss(self):
		self._losses.reset()

	def make_grad(self, tensor):
		return Variable(tensor, requires_grad=True)

	def run_forward(self, inputs, labels, one_hot_labels=False, num_classes=10):
		output_tensors = None
		count = 0

		for eachmodule in self.module_with_dependencies():
			# input_tensors
			if eachmodule.is_first():
				#input_tensors = inputs.to('cuda:' + str(self.module_with_dependencies()[0].rank()))
				input_tensors = inputs.cuda()
				eachmodule.input_tensors = self.make_grad(input_tensors)

			elif eachmodule.rank() != eachmodule.prev_rank():
				# receive
				# set tensor shapes temperately!
				input1 = torch.zeros([100, 1024, 2, 2],dtype=torch.float32).cuda()
				input2 = torch.zeros([100, 512, 1, 1],dtype=torch.float32).cuda()
				dist.broadcast(tensor=input1, src=0, group=self.process_groups[0][1])
				dist.broadcast(tensor=input2, src=0, group=self.process_groups[0][1])
				eachmodule.input_tensors = (self.make_grad(input1), self.make_grad(input2))
			else:
				input_tensors = output_tensors
				if isinstance(input_tensors, torch.Tensor):
					eachmodule.input_tensors = self.make_grad(input_tensors)
				else:
					eachmodule.input_tensors = tuple([self.make_grad(each) for each in input_tensors])

			#inference
			if eachmodule.is_first():
				output_tensors = eachmodule.module()(eachmodule.input_tensors)
				eachmodule.output_tensors = output_tensors
			elif eachmodule.is_last():
				if not one_hot_labels:
					# make labels one_hot
					labels = torch.sparse.torch.eye(num_classes).index_select(0, labels).cuda()			
				output_tensors = eachmodule.module()(eachmodule.input_tensors, labels)
				eachmodule.output_tensors = output_tensors
				self._loss = output_tensors
				self._losses.update(output_tensors)
			else:
				output_tensors = eachmodule.module()(*[eachtensor for eachtensor in eachmodule.input_tensors])
				eachmodule.output_tensors = output_tensors

			# send
			if eachmodule.next_rank() != None and eachmodule.next_rank() != eachmodule.rank():
				for each in output_tensors:
					#print(each.shape, each.dtype)
					dist.broadcast(tensor=each, src=0, group=self.process_groups[0][1])
					#print("forward send: ", time.time())
	
	def run_backward(self):
		#receive, if not last; send, if not first
		"""
		if self._loss != None:
			dist.broadcast(tensor=input1, src=0)
			print(input1)
		else:
			torch.autograd.backward(self.loss)
			dist.broadcast(tensor=input1, src=0)
		output_tensors = self.loss
		input_tensors = None
		grad_tensors = None

		torch.autograd.backward(output_tensors)
		"""
		if self.optimizer is None:
			return
		self.optimizer.zero_grad()

		last_module = None
		for each in reversed(self._module_with_dependencies):
			# the first module to backward
			if last_module is None:
				# if is the last module of the whole model, don't need to receive tensors
				if self._loss is not None:
					torch.autograd.backward(each.output_tensors)
				else:
					output1 = torch.zeros([100, 1024, 2, 2],dtype=torch.float32).cuda()
					output2 = torch.zeros([100, 512, 1, 1],dtype=torch.float32).cuda()
					dist.broadcast(tensor=output1, src=each.next_rank(), group=self.process_groups[0][1])
					dist.broadcast(tensor=output2, src=each.next_rank(), group=self.process_groups[0][1])
					torch.autograd.backward(each.output_tensors[0], 
						grad_tensors=output1, retain_graph=True)
					torch.autograd.backward(each.output_tensors[1], 
						grad_tensors=output2)
			else:
				if isinstance(each.output_tensors, torch.Tensor):
					torch.autograd.backward(each.output_tensors,
						grad_tensors=last_module.input_tensors.grad)
				else:
					for i in range(len(each.output_tensors)):
						torch.autograd.backward(each.output_tensors[i], 
							grad_tensors=last_module.input_tensors[i].grad, 
							retain_graph=True)
			last_module = each
		if last_module is not None and not last_module.is_first():
			if isinstance(last_module.output_tensors, torch.Tensor):
				dist.broadcast(tensor=last_module.input_tensors.grad, src=self.rank, group=self.process_groups[0][1])
			else:
				for eachtensor in last_module.input_tensors:
					dist.broadcast(tensor=eachtensor.grad, src=self.rank, group=self.process_groups[0][1])

		self.optimizer.step()


	def scale(self, rank, config):
		old_module_to_stage_map = self.module_to_stage_map
		old_stage_to_rank_map = self.stage_to_rank_map
		old_rank = self.rank
		old_module_with_dependencies = self._module_with_dependencies
		self._module_with_dependencies = []
		self.config = config
		json_config_file = json.load(open(config, 'r'))
		self.module_to_stage_map = json_config_file.get("module_to_stage_map", None)
		self.stage_to_rank_map = json_config_file.get("stage_to_rank_map", None)
		self.rank = int(rank)
		self._parameters = []
		self._loss = None
		self._losses = AverageMeter()

		del self.optimizer
		self.optimizer = None

		local_index = 0
		global_index = 0
		for (stage, inputs, outputs) in self.model:
			# if this stage is going away, send the parameters and find its old local_index	
			# old_rank and rank: the rank in the model
			# global_rank: the rank in all machines
			# default: old_rank = rank = global_rank	
			if old_stage_to_rank_map[old_module_to_stage_map[global_index]] == old_rank:
				if self.stage_to_rank_map[self.module_to_stage_map[global_index]] != self.rank:
					print("sending parameters in global_index: ", global_index)
					print("local_index: ", local_index, stage)
					print("dst = ", self.stage_to_rank_map[self.module_to_stage_map[global_index]])
					print("process group id: ", self.stage_to_rank_map[self.module_to_stage_map[global_index]])
					print("%d tensors" % (len(old_module_with_dependencies[local_index].module().state_dict())))
					for k, v in old_module_with_dependencies[local_index].module().state_dict().items():
						dist.broadcast(v, src=self.rank, group=self.process_groups[0][1])
				local_index += 1
			global_index += 1

		local_index = 0
		global_index = 0

		for (stage, inputs, outputs) in self.model:
			if self.stage_to_rank_map[self.module_to_stage_map[global_index]] != self.rank:
				global_index += 1
				continue
			# if a new stage is joining, receive parameters.
			# remove old self._module_with_dependencies
			if global_index == 0:
				prev_rank = None
				is_first = True
			else:
				prev_rank = self.stage_to_rank_map[self.module_to_stage_map[global_index - 1]]
				is_first = False
			if global_index == len(self.module_to_stage_map) - 1:
				is_last = True
				next_rank = None
			else:
				is_last = False
				next_rank = self.stage_to_rank_map[self.module_to_stage_map[global_index + 1]]
			
			self._module_with_dependencies.append(ModuleWithDependencies(stage(), 
				self.rank, prev_rank=prev_rank, next_rank=next_rank, 
				is_first=is_first, is_last=is_last))
			
			local_index += 1
			global_index += 1

		old_local_index, local_index = -1, -1
		global_index = 0
		for (stage, inputs, outputs) in self.model:
			if old_stage_to_rank_map[old_module_to_stage_map[global_index]] == old_rank:
				old_local_index += 1
			if self.stage_to_rank_map[self.module_to_stage_map[global_index]] == self.rank:
				local_index += 1
				if old_stage_to_rank_map[old_module_to_stage_map[global_index]] == old_rank:
					#in old one and in new one
					self._module_with_dependencies[local_index].module().load_state_dict(
						old_module_with_dependencies[old_local_index].module().state_dict())
				else:
					print("reveiveing parameters in global_index: ", global_index)
					print("local_index: ", local_index, stage)
					print("src = ", old_stage_to_rank_map[old_module_to_stage_map[global_index]])
					print("process group id: ", old_stage_to_rank_map[old_module_to_stage_map[global_index]])
					print("%d tensors" % (len(self._module_with_dependencies[local_index].module().state_dict())))
					for k, v in self._module_with_dependencies[local_index].module().state_dict().items():
						dist.broadcast(v, src=old_stage_to_rank_map[old_module_to_stage_map[global_index]],
							group=self.process_groups[0][1])
			global_index += 1

		del old_module_with_dependencies

		print(len(self._module_with_dependencies))
		for each in self._module_with_dependencies:
			print(each.prev_rank(), each.rank(), each.next_rank(), each.is_first(), each.is_last())

		for i in range(len(self._module_with_dependencies)):
			self._parameters.extend(self._module_with_dependencies[i].module().parameters())
		#print(self._parameters)
		
		if len(self._parameters) > 0:
			self.optimizer = torch.optim.SGD(self._parameters,lr=0.001)






