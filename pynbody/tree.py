"""

tree
====

Implements merger tree functions. If you have supported merger
tree files on disk or a merger tree tool installed and correctly 
configured, you can access a merger tree through f.tree() where f 
is a halo catalogue object.

"""

from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
from . import halo
import weakref
import logging
import os.path
import sys
import json
import glob
import re
import copy
from copy import deepcopy
from . import util, config, config_parser
import subprocess
from . import halo

logger = logging.getLogger("pynbody.tree")


try:
	from StringIO import StringIO as BytesIO
except ImportError:
	from io import BytesIO

import uuid

"""
   Python 2/3 Tree Implementation
   adapted from treelib https://github.com/caesar0301/treelib
"""

# ----------------------------#
# General tree class #
#-----------------------------#


class NodeIDAbsentError(Exception):
	"""Exception throwed if a node's identifier is unknown"""
	pass


class NodePropertyAbsentError(Exception):
	"""Exception throwed if a node's data property is not specified"""
	pass


class MultipleRootError(Exception):
	"""Exception throwed if more than one root exists in a tree."""
	pass


class DuplicatedNodeIdError(Exception):
	"""Exception throwed if an identifier already exists in a tree."""
	pass


class LinkPastRootNodeError(Exception):
	"""
	Exception throwed in Tree.link_past_node() if one attempts
	to "link past" the root node of a tree.
	"""
	pass


class InvalidLevelNumber(Exception):
	pass


class LoopError(Exception):
	"""
	Exception thrown if trying to move node B to node A's position
	while A is B's ancestor.
	"""
	pass


def python_2_unicode_compatible(klass):
	"""
	(slightly modified from :
		http://django.readthedocs.org/en/latest/_modules/django/utils/encoding.html)

	A decorator that defines __unicode__ and __str__ methods under Python 2.
	Under Python 3 it does nothing.

	To support Python 2 and 3 with a single code base, define a __str__ method
	returning text and apply this decorator to the class.
	"""
	if sys.version_info[0] == 2:
		if '__str__' not in klass.__dict__:
			raise ValueError("@python_2_unicode_compatible cannot be applied "
							 "to %s because it doesn't define __str__()." %
							 klass.__name__)
		klass.__unicode__ = klass.__str__
		klass.__str__ = lambda self: self.__unicode__().encode('utf-8')
	return klass

@python_2_unicode_compatible
class Tree(object):
	"""Generic Tree object
	Tree objects are made of Node(s) stored in _nodes dictionary.
	"""

	#: ROOT, DEPTH, WIDTH, ZIGZAG constants :
	(ROOT, DEPTH, WIDTH, ZIGZAG) = list(range(4))

	def __contains__(self, identifier):
		"""Return a list of the nodes'identifiers matching the
		identifier argument.
		"""
		return [node for node in self._nodes
				if node == identifier]

	def __init__(self, tree=None, deep=False):
		"""Initiate a new tree or copy another tree with a shallow or
		deep copy.
		"""

		#: dictionary, identifier: Node object
		self._nodes = {}

		#: identifier of the root node
		self.root = None

		if tree is not None:
			self.root = tree.root

			if deep:
				for nid in tree._nodes:
					self._nodes[nid] = deepcopy(tree._nodes[nid])
			else:
				self._nodes = tree._nodes

	def __getitem__(self, key):
		"""Return _nodes[key]"""
		try:
			return self._nodes[key]
		except KeyError:
			raise NodeIDAbsentError("Node '%s' is not in the tree" % key)

	def __len__(self):
		"""Return len(_nodes)"""
		return len(self._nodes)

	def __setitem__(self, key, item):
		"""Set _nodes[key]"""
		self._nodes.update({key: item})

	def __str__(self):
		self.reader = ""

		def write(line):
			self.reader += line.decode('utf-8') + "\n"

		self.__print_backend(func=write)
		return self.reader

	def __print_backend(self, nid=None, level=ROOT, idhidden=True, tree_filter=None,
					   key=None, reverse=False, line_type='ascii-ex',
					   data_property=None, func=print):
		"""
		Another implementation of printing tree using Stack
		Print tree structure in hierarchy style.

		For example:
			Root
			|___ C01
			|    |___ C11
			|         |___ C111
			|         |___ C112
			|___ C02
			|___ C03
			|    |___ C31

		A more elegant way to achieve this function using Stack
		structure, for constructing the Nodes Stack push and pop nodes
		with additional level info.

		UPDATE: the @key @reverse is present to sort node at each
		level.
		"""
		# Factory for proper get_label() function
		if data_property:
			if idhidden:
				def get_label(node):
					return getattr(node.data, data_property, node.data._get_property(data_property))
			else:
				def get_label(node):
					return "%s[%s]" % (getattr(node.data, data_property, node.data._get_property(data_property)), node.identifier)
		else:
			if idhidden:
				def get_label(node):
					return node.tag
			else:
				def get_label(node):
					return "%s[%s]" % (node.tag, node.identifier)

		# legacy ordering
		if key is None:
			def key(node):
				return node

		# iter with func
		for pre, node in self.__get(nid, level, tree_filter, key, reverse,
									line_type):
			label = get_label(node)
			func('{0}{1}'.format(pre, label).encode('utf-8'))

	def __get(self, nid, level, tree_filter_, key, reverse, line_type):
		# default tree_filter
		if tree_filter_ is None:
			def tree_filter_(node):
				return True

		# render characters
		dt = {
			'ascii': ('|', '|-- ', '+-- '),
			'ascii-ex': ('\u2502', '\u251c\u2500\u2500 ', '\u2514\u2500\u2500 '),
			'ascii-exr': ('\u2502', '\u251c\u2500\u2500 ', '\u2570\u2500\u2500 '),
			'ascii-em': ('\u2551', '\u2560\u2550\u2550 ', '\u255a\u2550\u2550 '),
			'ascii-emv': ('\u2551', '\u255f\u2500\u2500 ', '\u2559\u2500\u2500 '),
			'ascii-emh': ('\u2502', '\u255e\u2550\u2550 ', '\u2558\u2550\u2550 '),
		}[line_type]

		return self.__get_iter(nid, level, tree_filter_, key, reverse, dt, [])

	def __get_iter(self, nid, level, tree_filter_, key, reverse, dt, is_last):
		dt_vline, dt_line_box, dt_line_cor = dt
		leading = ''
		lasting = dt_line_box

		nid = self.root if (nid is None) else nid
		if not self.contains(nid):
			raise NodeIDAbsentError("Node '%s' is not in the tree" % nid)

		node = self[nid]

		if level == self.ROOT:
			yield "", node
		else:
			leading = ''.join(map(lambda x: dt_vline + ' ' * 3
								  if not x else ' ' * 4, is_last[0:-1]))
			lasting = dt_line_cor if is_last[-1] else dt_line_box
			yield leading + lasting, node

		if tree_filter_(node) and node.expanded:
			children = [self[i] for i in node.fpointer if tree_filter_(self[i])]
			idxlast = len(children)-1
			if key:
				children.sort(key=key, reverse=reverse)
			elif reverse:
				children = reversed(children)
			level += 1
			for idx, child in enumerate(children):
				is_last.append(idx == idxlast)
				for item in self.__get_iter(child.identifier, level, tree_filter_,
											key, reverse, dt, is_last):
					yield item
				is_last.pop()

	def __update_bpointer(self, nid, parent_id):
		"""set self[nid].bpointer"""
		self[nid].update_bpointer(parent_id)

	def __update_fpointer(self, nid, child_id, mode):
		if nid is None:
			return
		else:
			self[nid].update_fpointer(child_id, mode)

	def __real_true(self, p):
		return True

	def add_node(self, node, parent=None):
		"""
		Add a new node to tree.
		The 'node' parameter refers to an instance of Class::Node
		"""
		if not isinstance(node, Node):
			raise OSError("First parameter must be object of Class::Node.")

		if node.identifier in self._nodes:
			raise DuplicatedNodeIdError("Can't create node "
										"with ID '%s'" % node.identifier)

		pid = parent.identifier if isinstance(parent, Node) else parent

		if pid is None:
			if self.root is not None:
				raise MultipleRootError("A tree takes one root merely.")
			else:
				self.root = node.identifier
		elif not self.contains(pid):
			raise NodeIDAbsentError("Parent node '%s' "
									"is not in the tree" % pid)

		self._nodes.update({node.identifier: node})
		self.__update_fpointer(pid, node.identifier, Node.ADD)
		self.__update_bpointer(node.identifier, pid)

	def all_nodes(self):
		"""Return all nodes in a list"""
		return list(self._nodes.values())

	def all_nodes_itr(self):
		"""
		Returns all nodes in an iterator
		Added by William Rusnack
		"""
		return self._nodes.values()

	def children(self, nid):
		"""
		Return the children (Node) list of nid.
		Empty list is returned if nid does not exist
		"""
		return [self[i] for i in self.is_branch(nid)]

	def contains(self, nid):
		"""Check if the tree contains node of given id"""
		return True if nid in self._nodes else False

	def create_node(self, tag=None, identifier=None, parent=None, data=None):
		"""Create a child node for given @parent node."""
		node = Node(tag=tag, identifier=identifier, data=data)
		self.add_node(node, parent)
		return node

	def depth(self, node=None):
		"""
		Get the maximum level of this tree or the level of the given node

		@param node Node instance or identifier
		@return int
		@throw NodeIDAbsentError
		"""
		ret = 0
		if node is None:
			# Get maximum level of this tree
			leaves = self.leaves()
			for leave in leaves:
				level = self.level(leave.identifier)
				ret = level if level >= ret else ret
		else:
			# Get level of the given node
			if not isinstance(node, Node):
				nid = node
			else:
				nid = node.identifier
			if not self.contains(nid):
				raise NodeIDAbsentError("Node '%s' is not in the tree" % nid)
			ret = self.level(nid)
		return ret

	def expand_tree(self, nid=None, mode=DEPTH, tree_filter=None, key=None,
					reverse=False):
		"""
		Python generator. Loosly based on an algorithm from
		'Essential LISP' by John R. Anderson, Albert T. Corbett, and
		Brian J. Reiser, page 239-241

		UPDATE: the @tree_filter function is performed on Node object during
		traversing. In this manner, the traversing will not continue to
		following children of node whose condition does not pass the tree_filter.

		UPDATE: the @key and @reverse are present to sort nodes at each
		level.
		"""
		nid = self.root if (nid is None) else nid
		if not self.contains(nid):
			raise NodeIDAbsentError("Node '%s' is not in the tree" % nid)

		tree_filter = self.__real_true if (tree_filter is None) else tree_filter
		if tree_filter(self[nid]):
			yield nid
			queue = [self[i] for i in self[nid].fpointer if tree_filter(self[i])]
			if mode in [self.DEPTH, self.WIDTH]:
				queue.sort(key=key, reverse=reverse)
				while queue:
					yield queue[0].identifier
					expansion = [self[i] for i in queue[0].fpointer
								 if tree_filter(self[i])]
					expansion.sort(key=key, reverse=reverse)
					if mode is self.DEPTH:
						queue = expansion + queue[1:]  # depth-first
					elif mode is self.WIDTH:
						queue = queue[1:] + expansion  # width-first

			elif mode is self.ZIGZAG:
				# Suggested by Ilya Kuprik (ilya-spy@ynadex.ru).
				stack_fw = []
				queue.reverse()
				stack = stack_bw = queue
				direction = False
				while stack:
					expansion = [self[i] for i in stack[0].fpointer
								 if tree_filter(self[i])]
					yield stack.pop(0).identifier
					if direction:
						expansion.reverse()
						stack_bw = expansion + stack_bw
					else:
						stack_fw = expansion + stack_fw
					if not stack:
						direction = not direction
						stack = stack_fw if direction else stack_bw

	def tree_filter_nodes(self, func):
		"""
		tree_Filters all nodes by function
		function is passed one node as an argument and that node is included if function returns true
		returns a tree_filter iterator of the node in python 3 or a list of the nodes in python 2
		Added William Rusnack
		"""
		return tree_filter(func, self.all_nodes_itr())

	def get_node(self, nid):
		"""Return the node with `nid`. None returned if `nid` does not exist."""
		if nid is None or not self.contains(nid):
			return None
		return self._nodes[nid]

	def is_branch(self, nid):
		"""
		Return the children (ID) list of nid.
		Empty list is returned if nid does not exist
		"""
		if nid is None:
			raise OSError("First parameter can't be None")
		if not self.contains(nid):
			raise NodeIDAbsentError("Node '%s' is not in the tree" % nid)

		try:
			fpointer = self[nid].fpointer
		except KeyError:
			fpointer = []
		return fpointer

	def leaves(self, nid=None):
		"""Get leaves of the whole tree of a subtree."""
		leaves = []
		if nid is None:
			for node in self._nodes.values():
				if node.is_leaf():
					leaves.append(node)
		else:
			for node in self.expand_tree(nid):
				if self[node].is_leaf():
					leaves.append(self[node])
		return leaves

	def level(self, nid, tree_filter=None):
		"""
		Get the node level in this tree.
		The level is an integer starting with '0' at the root.
		In other words, the root lives at level '0';

		Update: @tree_filter params is added to calculate level passing
		exclusive nodes.
		"""
		return len([n for n in self.rsearch(nid, tree_filter)])-1

	def link_past_node(self, nid):
		"""
		Delete a node by linking past it.

		For example, if we have a -> b -> c and delete node b, we are left
		with a -> c
		"""
		if not self.contains(nid):
			raise NodeIDAbsentError("Node '%s' is not in the tree" % nid)
		if self.root == nid:
			raise LinkPastRootNodeError("Cannot link past the root node, "
										"delete it with remove_node()")
		# Get the parent of the node we are linking past
		parent = self[self[nid].bpointer]
		# Set the children of the node to the parent
		for child in self[nid].fpointer:
			self[child].update_bpointer(parent.identifier)
		# Link the children to the parent
		parent.fpointer += self[nid].fpointer
		# Delete the node
		parent.update_fpointer(nid, mode=parent.DELETE)
		del self._nodes[nid]

	def move_node(self, source, destination):
		"""
		Move a node indicated by @source parameter to be a child of
		@destination.
		"""
		if not self.contains(source) or not self.contains(destination):
			raise NodeIDAbsentError
		elif self.is_ancestor(source, destination):
			raise LoopError

		parent = self[source].bpointer
		self.__update_fpointer(parent, source, Node.DELETE)
		self.__update_fpointer(destination, source, Node.ADD)
		self.__update_bpointer(source, destination)

	def is_ancestor(self, ancestor, grandchild):
		parent = self[grandchild].bpointer
		child = grandchild
		while parent is not None:
			if parent == ancestor:
				return True
			else:
				child = self[child].bpointer
				parent = self[child].bpointer
		return False

	@property
	def nodes(self):
		"""Return a dict form of nodes in a tree: {id: node_instance}"""
		return self._nodes

	def parent(self, nid):
		"""Get parent node object of given id"""
		if not self.contains(nid):
			raise NodeIDAbsentError("Node '%s' is not in the tree" % nid)

		pid = self[nid].bpointer
		if pid is None or not self.contains(pid):
			return None

		return self[pid]

	def paste(self, nid, new_tree, deepcopy=False):
		"""
		Paste a @new_tree to the original one by linking the root
		of new tree to given node (nid).

		Update: add @deepcopy of pasted tree.
		"""
		assert isinstance(new_tree, Tree)
		if nid is None:
			raise OSError("First parameter can't be None")

		if not self.contains(nid):
			raise NodeIDAbsentError("Node '%s' is not in the tree" % nid)

		set_joint = set(new_tree._nodes) & set(self._nodes)  # joint keys
		if set_joint:
			# TODO: a deprecated routine is needed to avoid exception
			raise ValueError('Duplicated nodes %s exists.' % list(set_joint))

		if deepcopy:
			for node in new_tree._nodes:
				self._nodes.update({node.identifier: deepcopy(node)})
		else:
			self._nodes.update(new_tree._nodes)
		self.__update_fpointer(nid, new_tree.root, Node.ADD)
		self.__update_bpointer(new_tree.root, nid)

	def paths_to_leaves(self):
		"""
		Use this function to get the identifiers allowing to go from the root
		nodes to each leaf.
		Return a list of list of identifiers, root being not omitted.

		For example :
			Harry
			|___ Bill
			|___ Jane
			|    |___ Diane
			|         |___ George
			|              |___ Jill
			|         |___ Mary
			|    |___ Mark

		expected result :
		[['harry', 'jane', 'diane', 'mary'],
		 ['harry', 'jane', 'mark'],
		 ['harry', 'jane', 'diane', 'george', 'jill'],
		 ['harry', 'bill']]
		"""
		res = []

		for leaf in self.leaves():
			res.append([nid for nid in self.rsearch(leaf.identifier)][::-1])

		return res

	def remove_node(self, identifier):
		"""
		Remove a node indicated by 'identifier'; all the successors are
		removed as well.

		Return the number of removed nodes.
		"""
		removed = []
		if identifier is None:
			return 0

		if not self.contains(identifier):
			raise NodeIDAbsentError("Node '%s' "
									"is not in the tree" % identifier)

		parent = self[identifier].bpointer
		for id in self.expand_tree(identifier):
			# TODO: implementing this function as a recursive function:
			#       check if node has children
			#       true -> run remove_node with child_id
			#       no -> delete node
			removed.append(id)
		cnt = len(removed)
		for id in removed:
			del self._nodes[id]
		# Update its parent info
		self.__update_fpointer(parent, identifier, Node.DELETE)
		return cnt

	def remove_subtree(self, nid):
		"""
		Return a subtree deleted from this tree. If nid is None, an
		empty tree is returned.
		For the original tree, this method is similar to
		`remove_node(self,nid)`, because given node and its children
		are removed from the original tree in both methods.
		For the returned value and performance, these two methods are
		different:

			`remove_node` returns the number of deleted nodes;
			`remove_subtree` returns a subtree of deleted nodes;

		You are always suggested to use `remove_node` if your only to
		delete nodes from a tree, as the other one need memory
		allocation to store the new tree.
		"""
		st = Tree()
		if nid is None:
			return st

		if not self.contains(nid):
			raise NodeIDAbsentError("Node '%s' is not in the tree" % nid)
		st.root = nid

		parent = self[nid].bpointer
		self[nid].bpointer = None  # reset root parent for the new tree
		removed = []
		for id in self.expand_tree(nid):
			removed.append(id)
		for id in removed:
			st._nodes.update({id: self._nodes.pop(id)})
		# Update its parent info
		self.__update_fpointer(parent, nid, Node.DELETE)
		return st

	def rsearch(self, nid, tree_filter=None):
		"""
		Traverse the tree branch along the branch from nid to its
		ancestors (until root).
		"""
		if nid is None:
			return

		if not self.contains(nid):
			raise NodeIDAbsentError("Node '%s' is not in the tree" % nid)

		tree_filter = (self.__real_true) if (tree_filter is None) else tree_filter

		current = nid
		while current is not None:
			if tree_filter(self[current]):
				yield current
			# subtree() hasn't update the bpointer
			current = self[current].bpointer if self.root != current else None

	def save2file(self, filename, nid=None, level=ROOT, idhidden=True,
				  tree_filter=None, key=None, reverse=False, line_type='ascii-ex', data_property=None):
		"""Update 20/05/13: Save tree into file for offline analysis"""
		def _write_line(line, f):
			f.write(line + b'\n')

		handler = lambda x: _write_line(x, open(filename, 'ab'))

		self.__print_backend(nid, level, idhidden, tree_filter,
			key, reverse, line_type, data_property, func=handler)

	def show(self, nid=None, level=ROOT, idhidden=True, tree_filter=None,
			 key=None, reverse=False, line_type='ascii-ex', data_property=None):
		self.reader = ""

		def write(line):
			self.reader += line.decode('utf-8') + "\n"

		try:
			self.__print_backend(nid, level, idhidden, tree_filter,
				key, reverse, line_type, data_property, func=write)
		except NodeIDAbsentError:
			print('Tree is empty')

		print(self.reader)#.encode('utf-8'))

	def siblings(self, nid):
		"""
		Return the siblings of given @nid.

		If @nid is root or there are no siblings, an empty list is returned.
		"""
		siblings = []

		if nid != self.root:
			pid = self[nid].bpointer
			siblings = [self[i] for i in self[pid].fpointer if i != nid]

		return siblings

	def size(self, level=None):
		"""
		Get the number of nodes of the whole tree if @level is not
		given. Otherwise, the total number of nodes at specific level
		is returned.

		@param level The level number in the tree. It must be between
		[0, tree.depth].

		Otherwise, InvalidLevelNumber exception will be raised.
		"""
		if level is None:
			return len(self._nodes)
		else:
			try:
				level = int(level)
				return len([node for node in self.all_nodes_itr() if self.level(node.identifier) == level])
			except:
				raise TypeError("level should be an integer instead of '%s'" % type(level))
			

	def subtree(self, nid):
		"""
		Return a shallow COPY of subtree with nid being the new root.
		If nid is None, return an empty tree.
		If you are looking for a deepcopy, please create a new tree
		with this shallow copy,

		e.g.
			new_tree = Tree(t.subtree(t.root), deep=True)

		This line creates a deep copy of the entire tree.
		"""
		st = Tree()
		if nid is None:
			return st

		if not self.contains(nid):
			raise NodeIDAbsentError("Node '%s' is not in the tree" % nid)

		st.root = nid
		for node_n in self.expand_tree(nid):
			st._nodes.update({self[node_n].identifier: self[node_n]})
		return st

	def to_dict(self, nid=None, key=None, sort=True, reverse=False, with_data=False):
		"""transform self into a dict"""

		nid = self.root if (nid is None) else nid
		ntag = self[nid].tag
		tree_dict = {ntag: {"children": []}}
		if with_data:
			tree_dict[ntag]["data"] = self[nid].data

		if self[nid].expanded:
			queue = [self[i] for i in self[nid].fpointer]
			key = (lambda x: x) if (key is None) else key
			if sort:
				queue.sort(key=key, reverse=reverse)

			for elem in queue:
				tree_dict[ntag]["children"].append(
					self.to_dict(elem.identifier, with_data=with_data, sort=sort, reverse=reverse))
			if len(tree_dict[ntag]["children"]) == 0:
				tree_dict = self[nid].tag if not with_data else \
							{ntag: {"data":self[nid].data}}
			return tree_dict

	def to_json(self, with_data=False, sort=True, reverse=False):
		"""Return the json string corresponding to self"""
		return json.dumps(self.to_dict(with_data=with_data, sort=sort, reverse=reverse))


# ----------------------------#
# General Node class #
#-----------------------------#

class Node(object):
	"""
	Nodes are elementary objects which are stored a `_nodes` dictionary of a Tree.
	Use `data` attribute to store node-specific data.

	adapted from treelib https://github.com/caesar0301/treelib
	"""

	#: ADD, DELETE, INSERT constants :
	(ADD, DELETE, INSERT) = list(range(3))

	def __init__(self, tag=None, identifier=None, expanded=True, data=None):
		"""Create a new Node object to be placed inside a Tree object"""

		#: if given as a parameter, must be unique
		self._identifier = None
		self._set_identifier(identifier)

		#: None or something else
		#: if None, self._identifier will be set to the identifier's value.
		if tag is None:
			self._tag = self._identifier
		else:
			self._tag = tag

		#: boolean
		self.expanded = expanded

		#: identifier of the parent's node :
		self._bpointer = None
		#: identifier(s) of the soons' node(s) :
		self._fpointer = list()

		#: None or whatever given as a parameter
		self.data = data

	def __lt__(self, other):
		return self.tag < other.tag

	def _set_identifier(self, nid):
		"""Initialize self._set_identifier"""
		if nid is None:
			self._identifier = str(uuid.uuid1())
		else:
			self._identifier = nid

	@property
	def bpointer(self):
		"""Return the value of `_bpointer`."""
		return self._bpointer

	@bpointer.setter
	def bpointer(self, nid):
		"""Set the value of `_bpointer`."""
		if nid is not None:
			self._bpointer = nid
		else:
			# print("WARNING: the bpointer of node %s " \
			#      "is set to None" % self._identifier)
			self._bpointer = None

	@property
	def fpointer(self):
		"""Return the value of `_fpointer`."""
		return self._fpointer

	@fpointer.setter
	def fpointer(self, value):
		"""Set the value of `_fpointer`."""
		if value is None:
			self._fpointer = list()
		elif isinstance(value, list):
			self._fpointer = value
		elif isinstance(value, dict):
			self._fpointer = list(value.keys())
		elif isinstance(value, set):
			self._fpointer = list(value)
		else:  # TODO: add deprecated routine
			pass

	@property
	def identifier(self):
		"""Return the value of `_identifier`."""
		return self._identifier

	@identifier.setter
	def identifier(self, value):
		"""Set the value of `_identifier`."""
		if value is None:
			print("WARNING: node ID can not be None")
		else:
			self._set_identifier(value)

	def is_leaf(self):
		"""Return true if current node has no children."""
		if len(self.fpointer) == 0:
			return True
		else:
			return False

	def is_root(self):
		"""Return true if self has no parent, i.e. as root."""
		return self._bpointer is None

	@property
	def tag(self):
		"""Return the value of `_tag`."""
		return self._tag

	@tag.setter
	def tag(self, value):
		"""Set the value of `_tag`."""
		self._tag = value if value is not None else None

	def update_bpointer(self, nid):
		"""Update parent node."""
		self.bpointer = nid

	def update_fpointer(self, nid, mode=ADD):
		"""Update all children nodes."""
		if nid is None:
			return

		if mode is self.ADD:
			self._fpointer.append(nid)
		elif mode is self.DELETE:
			if nid in self._fpointer:
				self._fpointer.remove(nid)
		elif mode is self.INSERT:  # deprecate to ADD mode
			print("WARNING: INSERT is deprecated to ADD mode")
			self.update_fpointer(nid)

	def __repr__(self):
		name = self.__class__.__name__
		kwargs = [
			"tag=%r" % self.tag,
			"identifier=%r" % self.identifier,
			"data=%r" % self.data,
		]
		return "%s(%s)" % (name, ", ".join(kwargs))


class TreeHalo(halo.DummyHalo):

	"""
	Generic class representing a halo in a merger Tree.
	"""

	def __init__(self, halo_id, progenitor, shared_parts, filename, *args):

		super(TreeHalo, self).__init__(*args)
		# halo_id is in AHF convention thus add +1 to agree with pynbody convention
		self._halo_id = halo_id
		self._progenitor = progenitor
		self._shared_parts = shared_parts
		self._descriptor = "halo_" + str(halo_id+1)
		self.properties = copy.copy(self.properties)
		self.properties['halo_id'] = halo_id+1
		self.redshift = float(filename.split("z")[-1][0:5])

		f = util.open_(filename)
		if filename.split("z")[-2][-1] == ".":
			self.isnew = True
		else:
			self.isnew = False

		self._load_ahf_halo_data(filename,self._halo_id)

	def _get_property(self,prop):
		"""returns the property requested from the property dictionary."""
		return self.properties[prop]

	def _load_ahf_halo_data(self, filename, halo_id):
		f = util.open_(filename,"rt")
		# get all the property names from the first, commented line
		# remove (#)
		keys = [re.sub('\([0-9]*\)', '', field)
				for field in f.readline().split()]
		# provide translations
		for i, key in enumerate(keys):
			if self.isnew:
				if(key == '#npart'):
					keys[i] = 'npart'
			else:
				if(key == '#'):
					keys[i] = 'dumb'
			if(key == 'a'):
				keys[i] = 'a_axis'
			if(key == 'b'):
				keys[i] = 'b_axis'
			if(key == 'c'):
				keys[i] = 'c_axis'
			if(key == 'Mvir'):
				keys[i] = 'mass'

		if self.isnew:
			# fix for column 0 being a non-column in some versions of the AHF
			# output
			if keys[0] == '#':
				keys = keys[1:]

		for h, line in enumerate(f):
			if h == halo_id:
				values = [float(x) if '.' in x or 'e' in x or 'nan' in x else int(
				x) for x in line.split()]
				# XXX Unit issues!  AHF uses distances in Mpc/h, possibly masses as
				# well
				for i, key in enumerate(keys):
					if self.isnew:
						self.properties[key] = values[i]
					else:
						self.properties[key] = values[i - 1]
			else:
				pass
		f.close()

	@staticmethod
	def _can_load(self):
		return False

	@staticmethod
	def _can_run(self):
		return False


class AHFMergerTree(Tree):
	"""
	Class to handle merger tree catalogues produced by the 
	Amiga Halo Finder (AHF) merger tree tools. 

	For now it will just be a tree containing for every previous timestep
	the indices of progenitor halos. For future could add more information
	like mass, etc. or even hold the halo objects themselves.
	"""

	def __init__(self, parent_id, halo_catalogue, only_stat=True, dosort=None, **kwargs):
		"""Initialize an AHFMergerTree.

		**kwargs** :

		*ahf_basename*: specify the basename of the AHF merger tree
						files - the code will append 'mtree_idx',
						to this basename to load the catalog data.

		*only_stat*: specify that you only wish to collect the halo
					properties stored in the AHF_halos file and not
					worry about particle information

		*dosort*: specify if halo catalog should be sorted so that
				  halo 1 is the most massive halo, halo 2 the
				  second most massive and so on.

		"""

		import os.path
		if not self._can_load(halo_catalogue):
			self._run_ahf_mtree(halo_catalogue)

		self._ahfBasename = halo_catalogue._ahfBasename
		self._ahf_prefix = self._ahfBasename[:-18] #represents the prefix of the AHF files excluding the timesteps etc.

		self._root_id = parent_id
		self._root_id_ahf = parent_id-1

		# initialize the tree
		Tree.__init__(self)


		#fill the Node with the halo on which it was initialized
		self.create_node("halo "+str(parent_id), "root", data=TreeHalo(self._root_id_ahf, None, None, self._ahfBasename+'halos'))

		try:
			f = util.open_(self._ahfBasename + 'mtree')
		except IOError:
			raise IOError(
				"Merger tree files not found -- check the base name of tree data or try specifying tree files using the ahf_basename keyword")
		f.close()

		logger.info("AHFMergerTree filling tree")

		self._fill_ahf_tree(self._ahf_prefix, self._root_id_ahf)

		logger.info("AHFMergerTree loaded")

	def __getitem__(self,item):
		"""
		get the appropriate halo
		"""
		return super(AHFMergerTree,self).__getitem__(item)

	def __contains__(self, identifier):
		"""Return a list of the nodes'identifiers matching the
		identifier argument.
		"""
		return super(AHFMergerTree,self).__contains__(identifier)


	def __len__(self):
		"""Return len(_nodes)"""
		return super(AHFMergerTree,self).__len__()

	def __setitem__(self, key, item):
		"""Set _nodes[key]"""
		super(AHFMergerTree,self).__setitem__(key,item)

	def __str__(self):
		super(AHFMergerTree,self).__str__()


	def _load_ahf_mtree(self,filename,halo_id,particle_ratio=None):
		"""
		read in the AHF mtree file containing the indices of halos and its progenitors.
		"""
		progenitors = []
		shared_particles = []
		N_particles = []
		
		data = np.genfromtxt(filename,comments="#",dtype="int")
		i=0
		while i < len(data):
			if halo_id == data[i][0]:
				#if we found the halo of interest we load its progenitors
				for j in range(data[i][2]):
					idx = i+1+j
					if particle_ratio == None:
						progenitors.append(data[idx][1])
						shared_particles.append(data[idx][0])
					else:
						if (data[idx][0]/float(data[idx][2]) >= particle_ratio):
							progenitors.append(data[idx][1])
							shared_particles.append(data[idx][0])
				break
				# and then we do not need to do any more work.
			# if we did not find the halo of interest we directly jump to the next halo by adding N_progen to index i
			i+= data[i][2] + 1

		#f = util.open_(filename,"rt")
#		
#		#remove first two line, they are comments
#		for i, line in enumerate(f):
#			if i < 2:
#				continue
#			else: 
#				values = [int(x) for x in line.split()]
#				if halo_id == values[0]:
#					N_prog = values[2]
#					if j < values[2]: #this contains the number of progenitors
#						values = [int(x) for x in line.split()]
#						if particle_ratio == None:
#							progenitors.append(values[1])
#							shared_particles.append(values[0])
#						else:
#							if (values[0]/float(values[2]) >= particle_ratio):
#								progenitors.append(values[1])
#								shared_particles.append(values[0])
#				else:
#					if j  in range(values[2]): #this contains the number of progenitors
#						continue
			#i+=j+2
#		f.close()

		return progenitors, shared_particles

	def _fill_ahf_tree(self, prefix, halo_id, particle_ratio=0.75):

		tree_files = glob.glob(prefix+'*_mtree')
		tree_files.sort()
		ahf_files = glob.glob(prefix+'*_halos')
		ahf_files.sort()

		if len(tree_files)+1 == len(ahf_files):
			#check if there is a merger tree file to connect every successive output
			ahf_files = ahf_files[:-1]

			self._fill_tree(tree_files[::-1], ahf_files[::-1], halo_id, 1, "root", particle_ratio)
			# start from the redshift zero file, assuming we loaded the redshift zero snapshot to create the tree
		else:
			raise ValueError("Not all merger tree files are present! Please create all of them such that all successive snapshot files are connected.")
			
		return

	def _fill_tree(self, mtree_files, ahf_files, halo, level, parent, particle_ratio):

#		if level==1:
#			parent = "root"
#		else:
#			parent = "c_"+str(level-1)+"_"+str(halo)


		if len(mtree_files) == 1:
			# this is the stop condition for the recursive run
			progenitors, shared_particles = self._load_ahf_mtree(mtree_files[0], halo, particle_ratio)
			for i, h in enumerate(progenitors):
				tree_halo = TreeHalo(h, halo, shared_particles[i], ahf_files[0])
				if tree_halo.properties['hostHalo'] == -1:
					self.create_node("halo "+str(h), "c_"+str(level)+"_"+str(halo)+"_"+str(h), parent=parent, data=tree_halo)
		else:
			progenitors, shared_particles = self._load_ahf_mtree(mtree_files[0], halo, particle_ratio)
			for i, h in enumerate(progenitors):
				node_id = "c_"+str(level)+"_"+str(halo)+"_"+str(h)
				tree_halo = TreeHalo(h, halo, shared_particles[i], ahf_files[0])
				if tree_halo.properties['hostHalo'] == -1:
					try:
						self.create_node("halo "+str(h), node_id , parent=parent, data=tree_halo)
					except DuplicatedNodeIdError:
						print("Could not create node %s, duplicate node ID."%node_id)
						print("Most probably a progenitor with very few shared particles is tried to be traced.") 
						print("Try to rerun with particle_ratio bigger than %.2f"%particle_ratio)
					else:
						self._fill_tree(mtree_files[1::], ahf_files[1::], h, level+1, node_id, particle_ratio)	
		return


	def _read_main_branch(self, root_id, mtree_idx_files):
		"""reads in the AHFTreeHistory files to get the main branch of the merger tree."""

		main_branch_list = []
		main_branch_ids = []
		main_branch_list.append(root_id)
		main_branch_ids.append("root")
		halo = root_id
		for i, f in enumerate(mtree_idx_files):
			data = np.genfromtxt(f,comments="#",dtype="int")
			if len(data) > 2:
				for j in range(len(data)):
					try:
						if data[j][0] == halo:
							progenitor = data[j][1]
							node_id = "c_"+str(i+1)+"_"+str(halo)+"_"+str(progenitor)
							main_branch_list.append(str(progenitor))
							main_branch_ids.append(node_id)
							halo = progenitor
							break
					except:
						print("Sorry, something went wrong. Maybe the file format for this file %s is wrong."%f)
			else:
				if len(data) == 2:
					try:
						if len(data[0]) == 2:
							for j in range(len(data)):
								try:
									if data[j][0] == halo:
										progenitor = data[j][1]
										node_id = "c_"+str(i+1)+"_"+str(halo)+"_"+str(progenitor)
										main_branch_list.append(str(progenitor))
										main_branch_ids.append(node_id)
										halo = progenitor
										break
								except:
									print("Sorry, something went wrong. Maybe the file format for this file %s is wrong."%f)
					except:
						if isinstance(data[0],np.int64):
							if data[0] == halo:
								progenitor = data[1]
								node_id = "c_"+str(i+1)+"_"+str(halo)+"_"+str(progenitor)
								main_branch_list.append(str(progenitor))
								main_branch_ids.append(node_id)
								halo = progenitor
								break
						else:
							print("no progenitor found for this snapshot %s."%f)

		return main_branch_list, main_branch_ids

	def main_branch(self):
		all_branches = self.paths_to_leaves()
		mtree_idx_files = glob.glob(self._ahf_prefix+'*_mtree_idx')
		mtree_idx_files.sort()

		ahf_files = glob.glob(self._ahf_prefix+'*_halos')

		if len(mtree_idx_files)+1 == len(ahf_files):
			_main_branch, _main_branch_ids = self._read_main_branch(self._root_id_ahf,mtree_idx_files[::-1])
			if _main_branch_ids in all_branches:
				#new_tree = Tree(self.subtree(self.root), deep=True)
				_main_branch_list = []
				for id in _main_branch_ids:
					_main_branch_list.append(self.get_node(id).data)
					#try:
#						new_tree.remove_node(id)
#					except NodeIDAbsentError:
#						print("tried to remove node which was alredy removed...")

				self.main_branch_ids = _main_branch
				return _main_branch_list
			else:
				print("Main branch does not seem to be in the tree.")
		else:
			print("not all mtree_idx files present, please rerun the AHFTreeHistory tool.")







	def _plot_tree(self, ax, x_coord=0, y_coord=0, nid='root', color_property=None, **kwargs):
		"""
		Do the actual plotting of the whole tree.
		"""

		if len(self.children(nid)) >= 0:
			num = 2 * len(self.leaves(nid))
			coord_tmp = x_coord - num/2.

		for child in self.children(nid):
			cid = child.identifier
			num_child = 2 * len(self.leaves(cid))
			x_coord_child = coord_tmp + num_child/2.
			y_coord_child = -self.level(cid)
			child_mass = self.get_node(cid).data.properties['mass']

			if color_property != None:
				if isinstance(color_property, str):

					color_prop = self.get_node(cid).data.properties[color_property]
					c = ax.scatter(x_coord_child,y_coord_child,s=150*np.log10(child_mass/self.get_node('root').data.properties['mass']+1),zorder=10, c=np.log10(color_prop), **kwargs)
			
				else:
					raise ValueError("Color property must be string!")
			else:
				c = ax.scatter(x_coord_child,y_coord_child,s=150*np.log10(child_mass/self.get_node('root').data.properties['mass']+1),zorder=10, c='navy', **kwargs)
				
			lw = float("{0:.2f}".format(float(max(2*(child_mass/float(self.get_node('root').data.properties['mass'])),.2))))
			ax.plot([x_coord,x_coord_child], [y_coord,y_coord_child], color='black', linewidth=lw ,zorder=1)
			
			if self.get_node(cid).data.properties['M_star'] > 0:
				ax.scatter(x_coord_child,y_coord_child,s=100*np.log10(child_mass/self.get_node('root').data.properties['mass']+1),zorder=11,c='orange',marker='*',**kwargs)#np.log10(self.mass)
				
			self._plot_tree(ax,x_coord_child,y_coord_child,cid,color_property, **kwargs)

			coord_tmp += num_child


	def plot(self,filename=None, color_property=None, color_prop_label='', fancy=False, fancy_property=None, fancy_label='', **kwargs):
		"""
		Plot the merger tree where every node is represented by a circle connected with lines. Circle sizes represent 
		the virial masses in log scale. If color_property is set the circles are color coded by that property.
		If fancy is true at the right margin a plot for the main progenitor is created. With the fancy_property one can choose
		which property (from the AHF catalogue) to plot, e.g. if fancy_property='M_star' plots the stellar mass growth history 
		of the main progenitor.

		To Do: 	- if fancy_property is something like luminosity, read particle data and calculate that property for the main progenitor halos. 

				- build a filter to select only halos above a given mass...
		"""
		import matplotlib.pyplot as plt
		import matplotlib.gridspec as gridspec

		fig = plt.figure(figsize=(10,10))

		if fancy:
			gs = gridspec.GridSpec(2, 2, width_ratios=[5,1],height_ratios=[40,1],hspace=0.05,wspace=0.05)
			ax = plt.subplot(gs[0,0])

			axr = plt.subplot(gs[0,1])
			axr.spines['top'].set_visible(False)
			axr.spines['right'].set_visible(False)
			axr.xaxis.set_ticks_position('bottom')
			axr.yaxis.set_ticks_position('left')

		elif color_property != None:
			gs = gridspec.GridSpec(2, 1, height_ratios=[40,1],hspace=0.05,wspace=0.05)
			ax = plt.subplot(gs[0,0])
		else:
			ax = plt.subplot(111)


		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
		ax.spines['bottom'].set_visible(False)
		ax.xaxis.set_ticks_position('bottom')
		ax.yaxis.set_ticks_position('left')
		ax.set_ylabel(r'Redshift')

		#ax.set_ylim([-1,len(redshift)+1])
		#ax.set_xlim([0-self.count()/10.,2*self.count()*1.025])
		#ax.set_ylim(ax.get_ylim()[::-1])
		ax.axes.get_xaxis().set_visible(False)

		_main_branch = self.main_branch()

		if self.depth() < 10:
			yticks = np.linspace(0,self.depth(),self.depth()+1)
			red_labels = [x.redshift for x in _main_branch]
		else:
			_step = int(np.ceil(self.depth()/10.))
			yticks = np.linspace(0,self.depth(),self.depth()+1)[::_step]
			red_labels = [x.redshift for x in _main_branch[::_step]]
			
		ax.yaxis.set_ticks(-yticks)
		yticks_labels = [ '%.2f' % elem for elem, tmp in zip(red_labels,yticks) ]

		ax.set_yticklabels(yticks_labels)

		#xmin, xmax = ax.get_xaxis().get_view_interval()
		#ymin, ymax = ax.get_yaxis().get_view_interval()
	
		# width and depth of tree to scale the plot
		_depth = self.depth()
		_width = len(self.leaves())

		#here we could start with a function to plot
		# taking a node and applying itself to all children...

		_level = self.level('root')

		root_mass = self.get_node('root').data.properties['mass'] #returns virial mass in Msol

		if color_property != None:
			if isinstance(color_property, str):
				try:
					color_prop = self.get_node('root').data.properties[color_property]
				except: 
					raise ValueError("Property %s unknown!"%color_property)
				if "vmin" in kwargs:
					vmin = kwargs['vmin']
					if 'vmin' in kwargs: del kwargs['vmin']
				else:
					tmp = [np.log10(self.get_node(leave.identifier).data.properties[color_property]) if np.log10(self.get_node(leave.identifier).data.properties[color_property]) > 0 else 0. for leave in self.leaves()]
					vmin = np.min(tmp)
				if "vmax" in kwargs:
					vmax = kwargs['vmax']
					if 'vmax' in kwargs: del kwargs['vmax']
				else:
					vmax = np.log10(self.get_node('root').data.properties[color_property])

				c = ax.scatter(2*_width/2.,_level,s=150*np.log10(root_mass/root_mass+1),zorder=10,c=np.log10(color_prop),vmin=vmin,vmax=vmax,**kwargs)#np.log10(self.mass)
				
				self._plot_tree(ax,2*_width/2.,_level,'root',color_property, vmin=vmin, vmax=vmax, **kwargs)
			else:
				raise ValueError("Color property must be string!")

		else:
			ax.scatter(2*_width/2.,_level,s=150*np.log10(root_mass/root_mass+1),zorder=10, c='navy', **kwargs)
			
			self._plot_tree(ax,2*_width/2.,_level,'root',color_property, **kwargs)
		
		if self.get_node('root').data.properties['M_star'] > 0:
			ax.scatter(2*_width/2.,_level,s=100*np.log10(root_mass/root_mass+1),zorder=11,c='orange',marker='*',**kwargs)#np.log10(self.mass)
					
		if color_property != None:
			ax_help = plt.subplot(gs[1,0])
			cb = fig.colorbar(c,cax=ax_help,orientation='horizontal',ticks=np.linspace(vmin,vmax,5).tolist())

			cb.set_label(color_prop_label)
		
		if fancy:
			if fancy_property != None:
				if isinstance(fancy_property, str):
					try:
						fancy_prop = [x.properties[fancy_property] for x in _main_branch]
						y_pos = np.linspace(0,len(fancy_prop),len(fancy_prop))
						if 'cmap' in kwargs: del kwargs['cmap']
						axr.plot(np.log10(fancy_prop),-y_pos,'.r-',**kwargs)
						axr.set_xlabel(fancy_label)
						axr.set_yticklabels([])
						axr.set_xticks(axr.get_xticks()[::2])
					except: 
						raise ValueError("Property %s unknown!"%fancy_property)
				else:
					raise ValueError("Color property must be string!")

		if filename:
			fig.savefig(filename,bbox_inches='tight')


		return


	def _get_halo(self, i):
		# here we could implement something to get the particle data of any specific halo in the tree
		raise NotImplementedError

	@staticmethod
	def _can_load(halo_catalogue,**kwargs):
		for file in glob.glob(halo_catalogue._ahfBasename + 'mtree*'):
			if os.path.exists(file):
				return True
		return False

	def _run_ahf_mtree(self,halo_catalogue):

		mtree_tool = config_parser.get('AHFMergerTree', 'Path')

		if mtree_tool == 'None':
			for directory in os.environ["PATH"].split(os.pathsep):
				ahfs = glob.glob(os.path.join(directory, "AHF*"))
				for iahf, ahf in enumerate(ahfs):
					# if there are more AHF*'s than 1, it's not the last one, and
					# it's AHFstep, then continue, otherwise it's OK.
					if ((len(ahfs) > 1) & (iahf != len(ahfs) - 1) &
							(os.path.basename(ahf) == 'AHFstep')):
						continue
					else:
						mtree_tool = ahf
						break

		if not os.path.exists(mtree_tool):
			raise RuntimeError("Path to AHF (%s) is invalid" % mtree_tool)


		if os.path.exists(mtree_tool):
			# run it
			filenames = glob.glob(halo_catalogue._ahfBasename[:-18] + '*.z*_mtree')
			particlefiles = glob.glob(halo_catalogue._ahfBasename[:-18] + '.?????.z*_particles')
			if len(particlefiles) >= 2:
				num_files = len(particlefiles)
				if len(filenames) != num_files-1:
					# make sure there is a merger tree file present to connect all halo files.
					particlefiles.sort()
					# MergerTree input is number of mtree files N, then N mtree file names and finally N-1 prefixes for output files
					cmd = ''
					cmd1 = ''
					particlefiles = particlefiles[::-1]
					for i, f in enumerate(particlefiles[:-1]):
						cmd = cmd + f + '\n' 
						cmd1 = cmd1 + particlefiles[i][:-10] + '\n'
					cmd = cmd + particlefiles[-1] + '\n' + cmd1

					f = open('merger_tree.in', 'w')
					f.write(str(num_files)+'\n'+cmd)
					f.close()

					os.system("MergerTree < merger_tree.in")
					return
				else:
					print("Found %i snashots and %i merger tree files."%(num_files,len(filenames)))
					print("Nothing to do.")
					return
			else:
				print('Only less than 2 AHF files are present! If you want to run a merger tree tool, please make sure there is an AHF file for every snapshot.')
				return
				#here we maybe could automatically invoke a call to run AHF on missing snapshots
			

	@staticmethod
	def _can_run(halo_catalogue):
		if config_parser.getboolean('AHFMergerTree', 'AutoRun'):
			if config_parser.get('AHFMergerTree', 'Path') == 'None':
				for directory in os.environ["PATH"].split(os.pathsep):
					if (len(glob.glob(os.path.join(directory, "MergerTree"))) > 0):
						return True
			else:
				path = config_parser.get('AHFMergerTree', 'Path')
				return os.path.exists(path)
		return False



#--------------------------#
# AHF Merger Tree History class #
#--------------------------#

class AHFTreeHistory(halo.AHFCatalogue):

	"""
	Class to handle merger tree catalogues produced by the 
	Amiga Halo Finder (AHF) merger tree tools.
	"""

	# For now this class is unused. It would create catalogues similar to normal 
	# AHF catalogues but for the evolution of just one halo with the first line beein the redshift
	def __init__(self, halo_id, only_stat=True, **kwargs):
		"""Initialize a AHFMergerTreeHistory.

		**kwargs** :

		*ahf_basename*: specify the basename of the AHF merger tree
						files - the code will append 'mtree_idx',
						to this basename to load the catalog data.

		*only_stat*: specify that you only wish to collect the halo
					properties stored in the AHF_halos file and not
					worry about particle information

		"""

		self._pb_id = halo_id
		self._ahf_id = halo_id-1
		import os.path
		if not self._can_load(self.ahf_id):
			self._run_ahf_mtree_history(self.ahf_id)

		self._id = str(self._ahf_id)
		self._fielname = 'halo'+str((7-len(self._id)*[0]))+self._id+'.dat'
		self._only_stat = only_stat

		try:
			f = util.open_(self._filename)
		except IOError:
			raise IOError(
				"Merger Tree history not found")

		for i, l in enumerate(f):
			pass
		self._nhalos = i
		f.close()

		logger.info("AHFTreeHistory loading halos")
		super(AHFTreeHistory)._load_ahf_halos(self._filename)

		logger.info("AHFTreeHistory loaded")

	def __getitem__(self,item):
		"""
		get the appropriate halo if dosort is on
		"""
		return super(AHFTreeHistory,self).__getitem__(i)


	def _get_halo(self, i):
		return self._halos[i]

	@staticmethod
	def _can_load(halo_id,**kwargs):
		id = str(halo_id)
		for file in glob.glob('halo_'+str((7-len(id)*[0]))+id+'.dat'): #file names are like this halo_0003513.dat
			if os.path.exists(file):
				return True
		return False

	def _run_ahf_mtree_history(self, halo_id, *args, **kwargs):

		mtree_history = config_parser.get('ahfHaloHistory', 'Path')

		if mtree_history == 'None':
			for directory in os.environ["PATH"].split(os.pathsep):
				ahfs = glob.glob(os.path.join(directory, "AHF*"))
				for iahf, ahf in enumerate(ahfs):
					# if there are more AHF*'s than 1, it's not the last one, and
					# it's AHFstep, then continue, otherwise it's OK.
					if ((len(ahfs) > 1) & (iahf != len(ahfs) - 1) &
							(os.path.basename(ahf) == 'AHFstep')):
						continue
					else:
						mtree_history = ahf
						break

		if not os.path.exists(mtree_history):
			raise RuntimeError("Path to AHF (%s) is invalid" % mtree_history)


		if os.path.exists(mtree_history):
			# run it
			prefix = self._base().filename
			if redshifts == None:
				tmp_list = glob.glob(prefix+'*_halos') 
				prefix_list = [x[:-6] for x in tmp_list]
				redshift_list = [float(x[-15:-10]) for x in tmp_list]
			else:
				prefix_list = []
				np.savetxt('redshift.list', redshifts)
				for red in redshifts:
					prefix_list.append(glob.glob(prefix+'*'+red+'*_halos')[0][:-6])

			prefix_list.sort()
			redshift_list.sort()
			np.savetxt('prefix.list', prefix_list[::-1], fmt="%s")
			np.savetxt('redshift.list', redshift_list, fmt="%.3f")
			np.savetxt('halo_ids.list', halo_ids, fmt="%i")

			process = subprocess.Popen("ahfHaloHistory halo_ids.list prefix.list redshift.list", shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

			output = process.stdout.read()
		return


	@staticmethod
	def _can_run(halo_id):
		if config_parser.getboolean('ahfHaloHistory', 'AutoRun'):
			if config_parser.get('ahfHaloHistory', 'Path') == 'None':
				for directory in os.environ["PATH"].split(os.pathsep):
					if (len(glob.glob(os.path.join(directory, "AHF*"))) > 0):
						return True
			else:
				path = config_parser.get('ahfHaloHistory', 'Path')
				return os.path.exists(path)
		return False



def _get_merger_tree_classes():
	_tree_classes = [AHFMergerTree]

	return _tree_classes

def _get_merger_tree_history_classes():
	_tree_history_classes = [AHFTreeHistory]

	return _tree_history_classes



