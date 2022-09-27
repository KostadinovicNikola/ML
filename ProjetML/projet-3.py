# IMPORTATION
import numpy as np

# CLASSE ABSTRAITE
class Loss(object):
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass


class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        ## Annule gradient
        pass

    def forward(self, X):
        ## Calcule la passe forward
        pass

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters -= gradient_step*self._gradient

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        pass

#PARTIE 1 : LINEAIRE
class MSELoss(Loss):
    def forward(self, y, yhat):
        return np.linalg.norm( y-yhat, axis=1) ** 2

    def backward(self, y, yhat):
        return -2 * (y-yhat)

class Linear(Module):
    def __init__(self, input, output):
        self._input = input
        self._output = output
        self._parameters = 2 * ( np.random.random((self._input, self._output)) - 0.5 ) # valeur entre -1 et 1
        self.zero_grad()

    def zero_grad(self):
        self._gradient = np.zeros((self._input, self._output))

    def forward(self, X):
        assert X.shape[1] == self._input

        return np.dot( X, self._parameters)

    def update_parameters(self, gradient_step=1e-3):
        self._parameters -= gradient_step * self._gradient

    def backward_update_gradient(self, input, delta):
        assert input.shape[1] == self._input
        assert delta.shape[1] == self._output
        assert delta.shape[0] == input.shape[0] 
        
        self._gradient += np.dot( input.T, delta )

    def backward_delta(self, input, delta):
        assert input.shape[1] == self._input
        assert delta.shape[1] == self._output

        return np.dot( delta, self._parameters.T )

#PARTIE 2 : NON LINEAIRE
class TanH(Module):

    def forward(self, X):
        assert X.shape[1] == self._input

        return np.tanh(X)

    def backward_delta(self, input, delta):
        assert input.shape[1] == self._input
        assert delta.shape[1] == self._output

        return ( 1 - np.tanh(input) ** 2 ) * delta


class Sigmoide(Module):
    
    def forward(self, X):
        assert X.shape[1] == self._input

        return 1 / (1 + np.exp(-X))

    def backward_delta(self, input, delta):
        assert input.shape[1] == self._input
        assert delta.shape[1] == self._output

        val = 1 / (1 + np.exp(-input) )
        return delta * ( val * ( 1 - val) )


#PARTIE 3 : ENCAPSULAGE

class Sequentiel:
   
    def __init__(self, modules, loss):
        assert len(modules) > 0

        self._modules = modules
        self._loss = loss

    def forward(self, X): 
        list_forwards = [ self._modules[0].forward(X) ]
        for i in range(1, len(self._modules)):
            list_forwards.append( self._modules[i].forward( list_forwards[-1] ) )
        return np.array(list_forwards)
    
    def backward(self, list_forwards, delta):
        list_deltas =  [ delta ]
        for i in range(len(self._modules) - 1, 0, -1):
            list_deltas.append( self._modules[i].backward_delta( list_forwards[i-1], list_deltas[-1] ) )
        return np.array(list_deltas)


    def update_parameters(self, gradient_step=1e-3):
        for module in self._modules :
            module.update_parameters(gradient_step=gradient_step)
            module.zero_grad()


class Optim:
    
    def __init__(self, net, loss, eps):
        
        self._sequentiel = Sequentiel(net, loss)
        self._eps = eps
        
    def step(self, batch_x, batch_y):
        list_forwards = self._sequentiel.forward( batch_x )
        loss = self._sequentiel._loss.forward(batch_y,list_forwards[-1])
        delta = self._sequentiel._loss.backward_delta(batch_y,list_forwards[-1])
        list_deltas = self._sequentiel.backward(list_forwards, delta)
        self._sequentiel.update_parameters(self._eps)
        return loss
      




    