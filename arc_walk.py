import numpy as np

def cart_from_pol(r,theta):
    
    x = np.zeros(len(theta)+1)
    sin = np.sin(theta)
    cos = np.cos(theta)
    
    x[0] = r * cos[0]
    
    for i in range(1,len(theta)):
        x[i] = r * cos[i] * np.prod(sin[:i])
        
    x[-1] = r * np.prod(sin)
    
    return x


def pol_from_cart(x):

    x_sq=x**2
    r = np.sqrt(np.sum(x_sq))
    
    if len(x) > 1:
        theta = np.zeros(len(x)-1)
    
    for i,_ in enumerate(theta):
        theta[i] = np.arccos( x[i]/np.sqrt(np.sum(x_sq[i:])) )
        
    if x[-1] < 0:
        theta[-1] *= -1.0
        
    return (r,theta)


def shortest_angular_path(theta_start, theta_end, N_points):
    
    theta_end %= (2*np.pi)
    theta_start %= (2*np.pi)

    print('theta_end = ', theta_end)
    print('theta_start = ', theta_start)
        
    swap = theta_end < theta_start
    if swap:
        theta_end, theta_start = theta_start, theta_end
        
    theta = theta_end - theta_start
    print('theta = ', theta)            

    if theta <= np.pi:
        path = np.linspace(0, theta, N_points)
    else:
        path = np.linspace(theta, 2*np.pi, N_points)
        path %= (2*np.pi)
        path = np.flipud(path)
    
    path += theta_start
    path %= (2*np.pi)
    
    if swap:
        path = np.flipud(path)
    
    return(path)


def linspace_sph_pol(x, y, N_points):
    
    assert len(x) == len(y)
    
    r1,theta1 = pol_from_cart(x)
    r2,theta2 = pol_from_cart(y)
    
    r_path = np.linspace(r1, r2, N_points)

    theta_path = np.zeros([N_points, len(theta1)])
    for i in range(len(theta1)-1):
        theta_path[:,i] = np.linspace(theta1[i], theta2[i], N_points)
        
    theta_path[:,-1] = shortest_angular_path(theta1[-1],theta2[-1],N_points)
        
    cart_path = np.zeros([N_points, len(x)])
    for i in range(N_points):
        x_i = cart_from_pol(r_path[i],theta_path[i,:])
        cart_path[i,:] = x_i
    
    return cart_path
