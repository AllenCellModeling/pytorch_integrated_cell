from abc import ABC, abstractmethod 

class DataProviderABC(ABC):
    
    @abstractmethod
    def get_n_dat(self, train_or_test):
        pass
    
    @abstractmethod    
    def get_n_classes(self):
        pass
    
    @abstractmethod
    def get_image_paths(self, inds, train_or_test):
        pass
    
    @abstractmethod
    def get_images(self, inds, train_or_test):
        pass
    
    @abstractmethod
    def get_classes(self, inds, train_or_test):
        pass
    