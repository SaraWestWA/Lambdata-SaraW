from pdb import set_trace as breakpoint 
# attributes/properties
# invoke without parentheses

# behaviors

# methods
# invoke with parenthesis

class Dog():
    def __init__(self, name, age, breed, housebroke=True):
        self.name = name
        self.age = age
        self.housebroke = housebroke
        self.breed = breed
     
    def bark(self):
        print(f'{self.name} likes to bark!')


class Labrador(Dog):
     def __init__(self, name, age, breed, color, housebroke):
         super().__init__(self, name, age, breed, housebroke)
         self.color = color




if __name__ == '__main__':

    lucky = Dog('Lucky', 2, 'mutt')
    spike = Dog('Spike', 8, 'Boxer', False)
    happy = Labrador('Happy',0.5, 'Labrador', False, 'chocolate')
    breakpoint()

