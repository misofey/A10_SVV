

class Material:
    def __init__(self, name, properties) -> None:
        
        # Material name
        self.name = name
        
        # Ensure all Material properties have been provided
        assert all([property in properties.keys() for property in ['E', 'rho', 'alpha']]), f"Material {name} is badly defined. Ensure that you have specfied its Young's modulus 'E', density 'rho' and thermal expansion coefficient 'alpha'"

        # Assign material properties
        self.E     = properties['E']
        self.rho   = properties['rho']
        self.alpha = properties['alpha']

    def __getitem__(self, key):
        # Allow users to retrieve values from the Material's dictionary
        return self.__dict__[key]
    
    def __setitem__(self, key, value) -> None:
        # Allow users to assign values to items in the Mesh's dictionary
        self.__dict__[key] = value  
