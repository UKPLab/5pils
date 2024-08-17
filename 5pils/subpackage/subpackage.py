class SubPackageClass:
    def __init__(self, name):
        self.name = name
    
    def __str__(self):
        return f"SubPackage - {self.name}"
    
    def __repr__(self):
        return f"SubPackage - {self.name}"
    
    def __eq__(self, other):
        return self.name == other.name
    
    def something(self):
        return "SubPackage - something"