# Project Name: rAnPrivGP
# Author: rAnYKM (Jiayi Chen)
#
#          ___          ____       _       __________
#    _____/   |  ____  / __ \_____(_)   __/ ____/ __ \
#   / ___/ /| | / __ \/ /_/ / ___/ / | / / / __/ /_/ /
#  / /  / ___ |/ / / / ____/ /  / /| |/ / /_/ / ____/
# /_/  /_/  |_/_/ /_/_/   /_/  /_/ |___/\____/_/
#
# Script Name: ran_tree.py
# Date: May. 23, 2016

class rAnNode:
    def add_child(self, value):
        if self.isLeaf:
            self.isLeaf = False
        if value not in self.children:
            self.children[value] = rAnNode(value)

    def __init__(self, value):
        self.value = value
        self.isLeaf = True
        self.children = dict()

class rAnTree:
    """
    rAnTree is not a binary tree.
    It is a tree-like structure implemented by dictionary
    """

    def __recursive_build(self, node):
        sub_dict = dict()
        for v, n in node.children.iteritems():
            if n.isLeaf:
                sub_dict[v] = -1
            else:
                sub_dict[v] = self.__recursive_build(n)
        return sub_dict

    def __recursive_path(self, node, former_path, paths):
        for v, n in node.children.iteritems():
            if n.isLeaf:
                paths.append(former_path + ',' + n.value)
            else:
                self.__recursive_path(n, former_path + ',' + n.value, paths)

    def __recursive_root_path(self, node, paths):
        for v, n in node.children.iteritems():
            if n.isLeaf:
                paths.append(n.value)
            else:
                self.__recursive_path(n, n.value, paths)
        return paths

    def __recursive_show(self, node):
        print("node %s's children: " % str(node.value))
        for v, n in node.children.iteritems():
            print(v),
        print
        for v, n in node.children.iteritems():
            if not n.isLeaf:
                self.__recursive_show(n)

    def add(self, value, path_to_parent=None):
        if path_to_parent is None:
            path_to_parent = []
        current = self.root
        for p in path_to_parent:
            if p not in current.children:
                current.add_child(p)
            current = current.children[p]
        current.add_child(value)

    def add_path(self, path):
        current = self.root
        for p in path:
            if p not in current.children:
                current.add_child(p)
            current = current.children[p]

    def build_dict(self):
        current = self.root
        return self.__recursive_build(current)

    def display(self):
        self.__recursive_show(self.root)

    def get_paths(self):
        paths = list()
        self.__recursive_root_path(self.root, paths)
        return paths


    def __init__(self):
        self.root = rAnNode(0)

def main():
    t = rAnTree()
    t.add('jason')
    t.add('tom')
    t.add('peter', ['jason'])
    t.add('bob', ['jack', 'alice'])
    t.display()
    print t.get_paths()

if __name__ == '__main__':
    main()
