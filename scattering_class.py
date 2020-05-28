"""
This module contains the simulation code for our nuclear scattering
project for PHYS 3266. The code was written in collaboration with my 
teammates Kenji Bomar, Jon Braatz, Damon Griffin, and Matt Mandel, 
all of whom contributed greatly to our final project. 
"""

import numpy as np
import matplotlib.pyplot as plt

class Scattering:
    """
    This class contains methods used to simulate scattering data
    generated in kaon scattering experiments about a carbon-12 
    nucleus.
    """
    
    def __init__(self, data):
        """
        Initialize the simulated cross section data.
        """
        
        self.data = data
    
    def V_well(self, r, R, V_0):
        """
        This function defines a finite square well potential. 
        
        Inputs:
            r: [np array] distance from the nucleus
            R: [float] kaon nuclear radius
            V_0: [float] height of the well
            
        Output:
            [np array] finite square well mapping 
        """
        
        r[r < R] = V_0 # [J]
        r[r >= R] = 0.0 # [J]
        
        return r
        
    def V_yukawa(self, r, R, g):
        """
        This function defines the Yukawa potential.
        
        Inputs:
            r: [np array] distance from the nucleus
            R: [float] kaon nuclear radius
            g: [float] magnitude scaling constant 
            
        Output:
            [np array] Yukawa mapping
        """
        
        return (-g**2) * np.exp(-r/R) / r
        
    def V_ws(self, r, R, a, V_0):
        """
        This function defines the Woods-Saxon potential.
        
        Inputs:
            r: [np array] distance from the nucleus
            R: [float] kaon nuclear radius
            a: [float] kaon nuclear thickness
            V_0: [float] max potential value
            
        Output:
            [np array] Woods-Saxon mapping 
        """
        
        return -V_0 /(1 + np.exp((r-R)/a))
    
    def select_potential(self, name, params):
        """
        This method chooses between the 3 potentials.
        
        Inputs:
            name: [str] referencing the desired potential
                ('well', 'yukawa', 'ws')
            params: [list] containing function parameters
            
        Output:
            [lambda function] applied to desired potential
        """
        
        if name is 'well':
            return lambda r : self.V_well(r, params[0], params[1])
        elif name is 'yukawa':
            return lambda r : self.V_yukawa(r, params[0], params[1])
        elif name is 'ws':
            return lambda r : self.V_ws(r, params[0], params[1], params[2])
        else:
            print("Invalid potential name")
            
    
    def gauss_quad(self, N, lower, upper):
        """
        This code finds the zeros of the nth Legendre polynomial using
        Newton's method, starting from the approximation given in Abramowitz
        and Stegun 22.16.6.  The Legendre polynomial itself is evaluated
        using the recurrence relation given in Abramowitz and Stegun
        22.7.10.  The function has been checked against other sources for
        values of N up to 1000.
        
        Specifically, this method returns integration points and weights
        mapped to the interval [a,b], so that sum_i w[i]*f(x[i])
        is the Nth-order Gaussian approximation to the integral
        int_a^b f(x) dx. 
        
        Written by Mark Newman <mejn@umich.edu>, June 4, 2011.
        You may use, share, or modify this file freely. 
        
        Inputs:
            N: [int] number of steps
            lower: [float] lower intergral bound
            upper: [float] upper integral bound
            
        Outputs:
            xi: [float] function integration points
            wi: [float] function weights
        """
        
        # Initial approximation to roots of the Legendre polynomial
        a = np.linspace(3,4*N-1,N)/(4*N+2)
        x = np.cos(np.pi*a+1/(8*N*N*np.tan(a)))

        # Find roots using Newton's method
        epsilon = 1e-15
        delta = 1.0
        
        while delta > epsilon:
            p0 = np.ones(N,float)
            p1 = np.copy(x)
            for k in range(1,N):
                p0,p1 = p1,((2*k+1)*x*p1-k*p0)/(k+1)
            dp = (N+1)*(p0-x*p1)/(1-x*x)
            dx = p1/dp
            x -= dx
            delta = max(abs(dx))

        # Calculate the weights
        w = 2*(N+1)*(N+1)/(N*N*(1-x*x)*dp*dp)
        
        xi = 0.5*(upper-lower)*x+0.5*(upper+lower)
        wi = 0.5*(upper-lower)*w
        
        return xi, wi 
        
    def integrate(self, integrand):
        """
        This method numerically integrates a given integrand using Gaussian Quadrature.
        The integral is computed by summing the products of the coefficients and the function 
        evaluated at the calculated points. The integrand has to be evaluated at modified points 
        in order to integrate from 0 to ∞ in the Born Approximation.
        
        Input: 
            [function] to integrate
            
        Output:
            [float] integral value
        """
    
        xi, wi = self.gauss_quad(100,0,1)
        
        return np.sum(wi* integrand(xi/(1-xi)) / (1-xi)**2)
        
    def differential_cross_section(self, theta, V, R):
        """
        This method computes the differential cross section in the kaon collision using 
        the Born Approximation. 
        
        Inputs:
            theta: [float] angle of approach
            V: potential [function]
            R: [float] kaon nuclear radius
            
        Output:
            [float] Born-approximated differential cross section
        """
    
        c = 2.998e+8 # Speed of Light [m/s]
        hbar = 1.0545718e-34 # Reduced Planck's Constant[Js]
        m = 7.899e-11/(c**2) # carbon-12 mass  [kg]
        p = 1.282e-10/c   # kaon momentum [kg m/s] 
    
        q = (2*p/hbar)*np.sin(theta/2.0) # wavevector [m^-1]
        
        # x has been scaled by R so that gauss_quad will sample the correct points.
        integrand = lambda x : x * V(x*R) * np.sin(q*x*R)
    
        # Born Approximation modified to incorporate the effects of the scaling constant. 
        return (-m/(hbar * p * np.sin(theta/2)) * self.integrate(integrand))**2 * R**4
        
    def cross_section_array(self, theta_array, V, params):
        """
        This method computes the differential cross section for a range
        of angles.
        
        Inputs:
            theta_array: [np array] of angles
            V: [str] potential function
            params: [list] of potential function parameters
            
        Output:
            [np array] of differnetial cross sections for a range of angles
        """
            
        cross_sections = np.zeros([len(theta_array)])
    
        # Apply the cross section function to all given angles. 
        for i in range(len(theta_array)):
            cross_sections[i] = self.differential_cross_section(theta_array[i], 
                                                                self.select_potential(V, params), 
                                                                params[0])
        
        return cross_sections

    def sum_squared_error(self, calculated, experimental):
        """
        This method calculates the sum of squared error
        values.
        
        Inputs:
            calculated: [np array] of simulated cross sections
            experimental: [np array] of experimental cross sections
            
        Output:
            [float] sum of square errors
        """
        
        return ((calculated - experimental)**2).sum()

    def test_fit(self, V, params, data):
        """
        This method compares the experimental cross section data
        with the simulated cross section data.
        
        Input:
            V: [str] potential function
            params: [list] of potential function parameters
            data: [np array] of experimental cross sections
            
        Output:
            [float] sum of squared errors between the simulated data 
                    and the experimental data
        """
    
        # Angles and cross sections
        thetas = data[:,0]
        experimental = data[:,1]
    
        # Predicted cross sections 
        calculated = self.cross_section_array(thetas, V, params)        
        
        return self.sum_squared_error(calculated, experimental)
        
    def plot_experiment(self):
        """
        This method plots the experimental data.
        """
        
        thetas = self.data[:,0]
        cross_sections = self.data[:,1]*1.0e-31/0.572 # Convert to [m^2/rad]
        
        errors = self.data[:,2]*1.0e-31/0.572
        cross_minus = cross_sections - errors
        cross_plus = cross_sections + errors
        
        plt.plot(thetas, cross_sections, label = 'Data')
        plt.plot(thetas, cross_minus, 'r:', label = 'Error Bound')
        plt.plot(thetas, cross_plus, 'r:')
        plt.xlabel('Angle (°)')
        plt.ylabel('Cross Section ($m^2/rad$)')
        plt.title('K+ Carbon-12 Scattering Experimental Data')
        plt.legend()
        
    def two_parameter_annealing(self, V_name, data, param1, param2):
        """
        This method optimizes a given function with respect to 2 parameters by
        minimizing the sum of squared errors. This code was adapted from code in
        "Computational Physics" by Mark Newman, p.494-496. 
        
        Inputs:
            V_name: [str] potential function to optimize
            data: [np array] experimental data
            
        Outputs:
            param1, param2: [float] optimial potential function parameters
            changes: [int] number of times annealing moved closer to optima
            
        """
        
        # Define extent and initial parameter guesses
        Tmax = 10.
        Tmin = 1e-3
        tau = 10. 

        # Record the number of times the process moved closer to the minimum
        changes = 0

        # Angles and cross sections
        data[:,0] *= np.pi / 180
        data[:,1] *= 1.0e-31/0.572 # convert from mb/sr to SI units

        # Initial error to be minimized 
        error = self.test_fit(V_name, [param1, param2], data)*1e56 # Scale up to avoid precision problems.

        # Initialize parameters
        t = 0
        T = Tmax

        # Loop until the minimum is reached. 
        while T>Tmin:
    
            # Increment time and temp 
            t += 1
            T = Tmax*np.exp(-t/tau) # cooling schedule
    
            # Randomly decide whether to update the first or second parameter
            if np.random.random() > .5:
                new_param1 = param1 + (.5 - np.random.random())*param1/50 
                new_param2 = param2
            else:
                new_param1 = param1
                new_param2 = param2 + (.5 - np.random.random())*param2/50 
        
            # new error using the updated values 
            new_error = self.test_fit(V_name, [new_param1, new_param2], data)*1e56 
    
            # change in error
            deltaE = new_error - error
    
            # If the change is negative, we have moved closer to the minimum. 
            if deltaE < 0 :
                changes += 1
                param1 = new_param1
                param2 = new_param2
                error = new_error

        return param1, param2, changes

    def three_parameter_annealing(self, V_name, data, param1, param2, param3):
        """
        This method optimizes a given function with respect to 3 parameters by
        minimizing the sum of squared errors. This code is a modification of 
        the prior 2 parameter annealing method, itself modified from code in
        "Computational Physics" by Mark Newman, p.494-496.
        
        Inputs:
            V_name: [str] potential function to optimize
            data: [np array] experimental data
            
        Outputs:
            param1, param2, param3: [float] optimial potential function parameters
            changes: [int] number of times annealing moved closer to optima
            
        """
        # Define extent and initial guesses
        Tmax = 10.
        Tmin = 1e-3
        tau = 100.

        # Record the number of times the process moved closer to the minimum
        changes = 0

        # Angles and cross sections
        data[:,0] *= np.pi / 180
        data[:,1] *= 1.0e-31/0.572 # convert from mb/sr to SI units

        # Initial error to be minimized 
        error = self.test_fit(V_name, [param1, param2, param3], data)*1e56 # Scale up to avoid precision problems.

        # Initialize parameters
        t = 0
        T = Tmax

        # Loop until the minimum is reached
        while T>Tmin:
    
            # Increment time and temp
            t += 1
            T = Tmax*np.exp(-t/tau) # cooling schedule
    
            # Choose a random number for updating the parameters.
            num = np.random.random()
    
            # Randomly decide whether to update the first, second, or third parameter. 
            if num < 0.3333:
                new_param1 = param1 + (.5 - np.random.random())*param1/50
                new_param2 = param2
                new_param3 = param3
            elif num < 0.666:
                new_param1 = param1
                new_param2 = param2 + (.5 - np.random.random())*param2/50
                new_param3 = param3
            else:
                new_param1 = param1
                new_param2 = param2  
                new_param3 = param3 + (.5 - np.random.random())*param3/50
    
            # New error using the updated values 
            new_error = self.test_fit(V_name, [new_param1, new_param2, new_param3], data)*1e56 
    
            # Change in error
            deltaE = new_error - error
    
            # If the change is negative, we have moved closer to the minimum
            if deltaE < 0 :
                changes += 1
                param1 = new_param1
                param2 = new_param2
                param3 = new_param3
                error = new_error

        return param1, param2, param3, changes
