
# attributes/properties
# invoke without parentheses

# behaviors

# methods
# invoke with parenthesis

class Polo():
    def__init__(self, color, size, price, style='Normal Fit'):
        self.color = color
        self.size = size
        self. = price
        self.style = style
     
    def fold(self):
        print ('Folding the' {self.size.upper()} {self.color.upper()} 'polo here!')


if __name__ == '__main__':  
    
    polo = Polo('blue', 'large', 99.99)
    print(type(polo))
    print(polo.color, polo.size, polo.price)
    polo.fold()

    polo2 = Polo(color='yellow', size='small', price=69.99)
    print(polo.color, polo.size, polo.price)
    polo.fold()

    polo3 = Polo(color='green', size='medium', price=69.99, style='Fit Cut')
    print(polo.color, polo.size, polo.price)
    polo.fold()
