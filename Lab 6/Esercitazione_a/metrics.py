import sys
import numpy as np

np.random.seed(42)


class Metrics():
    def __init__ (self, classes: list[str], num_data: int) -> None:
        self.classes = classes
        self.num_classes = len(classes)
        self.real_y = np.random.randint(0, self.num_classes, num_data)
        self.pred_y = np.random.randint(0, self.num_classes, num_data)
        self.confusion_matrix = None
    
    def compute_confusion_matrix(self) -> None:
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
        for true_label, pred_label in zip(self.real_y, self.pred_y):
            self.confusion_matrix[true_label][pred_label] += 1
    
    def accuracy(self) -> float:
        if self.confusion_matrix == None:
            self.compute_confusion_matrix()
        total_samples = np.sum(self.confusion_matrix)
        correct_predictions = np.sum(np.diag(self.confusion_matrix))
        if total_samples > 0:
            accuracy = correct_predictions / total_samples
        else:
            accuracy = 0.0
        return accuracy
    
    def recall(self, class_id: int) -> float:
        if (self.__valid_class_id(class_id)):
            if self.confusion_matrix == None:
                self.compute_confusion_matrix()
            true_positives = self.confusion_matrix[class_id][class_id]
            actual_positives = np.sum(self.confusion_matrix[class_id, :])
            if actual_positives > 0:
                recall = true_positives / actual_positives
            else:
                recall = 0.0
            return recall
        else:
            sys.exit(-1)

    def precision(self, class_id: int) -> float:
        if (self.__valid_class_id(class_id)):
            if self.confusion_matrix == None:
                self.compute_confusion_matrix()
            true_positives = self.confusion_matrix[class_id][class_id]
            total_positives = np.sum(self.confusion_matrix[:, class_id])
            if total_positives > 0:
                precision = true_positives / total_positives
            else:
                precision = 0.0
            return precision
        else:
            sys.exit(-1)
    
    def f1_score (self, class_id: int) -> float:
        if (self.__valid_class_id(class_id)):
            if self.confusion_matrix == None:
                self.compute_confusion_matrix()
            precision = self.precision(class_id)
            recall = self.recall(class_id)
            if precision > 0 and recall > 0 :
                f1_score = 2*((precision*recall)/(precision+recall))
            else:
                f1_score = 0.0
            return f1_score
        else:
            sys.exit(-1)
    
    def support (self, class_id: int) -> int:
        return 0
    
    def report(self) -> None:
        return 0
    
    def __valid_class_id(self, class_id: int) -> bool:
        if class_id < self.num_classes and class_id > 0:
            return True
        else:
            return False
    

if __name__ == '__main__':

    mt = Metrics(['circle', 'square', 'triangle'], 20)
    mt.report
    mt.compute_confusion_matrix()