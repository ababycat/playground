class Node(object):
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BST(object):
    def __init__(self, root):
        self.root = Node(root)

    def insert(self, new_val):
        current = self.root
        while True:
            if current.value == new_val:
                return 
            elif current.value < new_val:
                node = current.left
                if node is None:
                    current.left = Node(new_val)
                    break
            elif current.value > new_val:
                node = current.right
                if node is None:
                    current.right = Node(new_val)
                    break
            current = node
        return 

    def search(self, find_val):
        current = self.root
        while current:
            if current.value == find_val:
                return True
            elif current.value < find_val:
                current = current.left
            elif current.value > find_val:
                current = current.right
        return False

    def print_tree(self):
        self.print_tree_node(self.root)
    
    def print_tree_node(self, root):
        if root:
            self.print_tree_node(root.right)
            print(root.value)
            self.print_tree_node(root.left)

# Set up tree
tree = BST(4)

# Insert elements
tree.insert(2)
tree.insert(1)
tree.insert(3)
tree.insert(5)

#
tree.print_tree()
# Check search
# Should be True
print tree.search(4)
# Should be False
print tree.search(6)