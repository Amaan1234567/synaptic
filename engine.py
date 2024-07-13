import math
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from graphviz import Digraph

# basic unit of data
class Value:
    def __init__(self,data,_children=(),_op='',label='') -> None:
        self.data = data
        self.grad = 0
        self._backward = lambda : None
        self._prev = set(_children)
        self._op = _op
        self.label=label

    def __repr__(self):
        return f"value(data={self.data})"
    
    def __add__(self,other):
        other=other if isinstance(other,Value) else Value(other)
        out = Value(self.data+other.data,(self,other),'+')
        def _backward():
            self.grad += 1.0*out.grad
            other.grad += 1.0*out.grad
        out._backward = _backward
        return out
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self,other):
        return self + (-other)
    
    def __mul__(self,other):
        other=other if isinstance(other,Value) else Value(other)
        out = Value(self.data*other.data,(self,other),'*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    def __rmul__(self,other):
        return self*other
    def __truediv__(self,other):
        return self*other**-1
    
    def __pow__(self,other):
        assert isinstance(other,(int,float))
        if(self.data==0.0 and other<0):
            out=Value(-999999999,(self,),f'**{other}')
        else:
            out = Value(self.data**other,(self,),f'**{other}')
        def _backward():
            self.grad +=other*self.data ** (other-1) * out.grad
        out._backward = _backward
        return out
    def exp(self):
        x = self.data
        #print(x)
        if(x<-20.4206807):
            out=Value(0.000000001,(self,),'exp')
        elif(x>90):
            out = Value(999999999,(self,),'exp')
        else:
            #print(x)
            out = Value(math.exp(x),(self,),'exp')

        def _backward():
            self.grad = out.data * out.grad
        out._backward = _backward
        return out

    def log(self,base=math.e):
        if(self.data<=0.0):
            out=Value(-math.inf,(self,),f"log with base {round(base,2)}")
        else:
            out = Value(math.log(self.data,base),(self,),f"log with base {round(base,2)}")
        def _backward():
            self.grad += (1/(out.data+0.000000001)) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x=self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        t = round(t,6)
        out = Value(t,(self,),'tanh')
        def _backward():
            self.grad += (1-t**2) * out.grad
        out._backward=_backward
        return out  
    
    def relu(self):
        if(self.data>0):
            out=Value(self.data,(self,),'relu')
        else:
            out=Value(0,(self,),'relu')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward=_backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
        
        build_topo(self)
        self.grad=1.0
        for node in reversed(topo):
            node._backward()


class Neuron:
    def __init__(self,nin) -> None:
        self.w = [Value(random.standard_normal(1)[0]) for _ in range(nin)]
        self.b = Value(random.standard_normal(1)[0])
    def __call__(self,x):
        # w*x+b
        act=self.b
        for wi,xi in zip(self.w,x):
            act+=(wi*xi)
        #print(act)
        out = act.relu()
        return out
    def parameters(self):
        return self.w + [self.b]
    
class Layer:
    def __init__(self,nin,nout) -> None:
        self.neurons = [Neuron(nin) for _ in range(nout)]
    def __call__(self,x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs)==1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
        

class MLP:
    def __init__(self,nin,nouts):
        sz = [nin]+nouts
        self.layers = [Layer(sz[i],sz[i+1]) for i in range(len(nouts))]

    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()] 

class SGD:
    @staticmethod
    def optmiser_step(model,learning_rate):
        assert isinstance(model,MLP), ValueError("argument given to SGD was not instance of MLP class")
        print("inside optimiser step")
        for param in model.parameters():
            param.data += -learning_rate*param.grad
            param.grad*=0.01


class Utils:
    @staticmethod
    def sigmoid(x):
        return Value(1)/(Value(1)+(-x).exp())

    @staticmethod
    def softmax(x):
        print("inside softmax")
        #print(x,len(x))
        assert len(x)!=0, ValueError("expected list of Values but got empty list")
        assert isinstance(x[0],Value), ValueError("please return the value object, not its data")

        sums=Value(0)
        maxi=0
        for data in x:
            maxi=max(maxi,data.data)
        #print("sum: ",sums)
        #print("maxi: ",maxi)
        maxi=Value(maxi)
        for data in x:
            sums+=(data-maxi).exp()
        
        for i,data in enumerate(x):
            #print("before adjustment: ",data)
            data.data-=maxi.data
            #print("after adjustment: ",data)
            x[i]=((data.exp())/sums)
            #print("x[i]: ",x[i])
        return x
    
    @staticmethod
    def cross_entropy(y_true,y_pred):
        print("inside cross_entropy")
        loss = Value(0)

        for i,y in enumerate(y_true):
            loss+=-y*(y_pred[i].log())
        return loss

    def trace(root):
        nodes, edges = set(), set()
        def build(v):
            if v not in nodes:
                nodes.add(v)
                for child in v._prev:
                    edges.add((child, v))
                    build(child)
        build(root)
        return nodes, edges
    @staticmethod
    def draw_dot(root, format='svg', rankdir='LR'):
        """
        format: png | svg | ...
        rankdir: TB (top to bottom graph) | LR (left to right)
        """
        assert rankdir in ['LR', 'TB']
        nodes, edges = Utils.trace(root)
        dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})
        
        for n in nodes:
            dot.node(name=str(id(n)), label = "{ %s | data %.4f | grad %.4f }" % (n.label,n.data, n.grad), shape='record')
            if n._op:
                dot.node(name=str(id(n)) + n._op, label=n._op)
                dot.edge(str(id(n)) + n._op, str(id(n)))
        
        for n1, n2 in edges:
            dot.edge(str(id(n1)), str(id(n2)) + n2._op)
        
        return dot
