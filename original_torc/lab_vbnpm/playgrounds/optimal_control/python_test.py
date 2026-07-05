class MyClass:
    def func(self, arg: int) -> str:
        return str(arg)
    
    def func(self, arg: str) -> int:
        return len(arg)

# Usage
obj = MyClass()
print(obj.func(10))    # Output: "10"
print(obj.func("hello"))   # Output: 5