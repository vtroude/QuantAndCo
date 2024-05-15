import numpy as np

from typing import Union, Tuple



#######################################################################################################################



def kalman_step(
                x: np.ndarray,
                y: np.ndarray,
                state: np.ndarray,
                covariance: np.ndarray,
                noise: np.ndarray,
                noise_obs: np.ndarray,
                rate: float = None
            ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Performs a single step of the Kalman filter algorithm.

    Args:
        x (np.ndarray): The state transition matrix with shape (K,N)
        y (np.ndarray): The observation matrix with shape (K,1)
        state (np.ndarray): The current state estimate with shape (N,1)
        covariance (np.ndarray): The current state covariance matrix with shape (N,N)
        noise (np.ndarray): The process noise covariance matrix with shape (N,N)
        noise_obs (np.ndarray): The observation noise covariance matrix with shape (K,K)
        rate (float, optional): The rate of adaptation for the noise covariance matrices. Defaults to None.

    Returns:
        tuple: A tuple containing the updated state estimate, updated state covariance matrix,
               updated process noise covariance matrix, and updated observation noise covariance matrix.
    """

    # Prediction step
    predicted_covariance = covariance + noise

    # Update step
    residual            = y - x.dot(state)                                                      # Residual Error (ex-ante)
    residual_covariance = x.dot(predicted_covariance.dot(x.T)) + noise_obs                      # Residual Covariance (ex-ante)
    
    # Kalman Gain (Filter)
    print(residual_covariance)
    kalman_gain         = predicted_covariance.dot(x.T.dot(np.linalg.inv(residual_covariance)))

    # Update state
    updated_state   = state + kalman_gain.dot(residual)                                                                 # New State (ex-post)
    updated_covariance = (np.eye(np.shape(predicted_covariance)[0]) - kalman_gain.dot(x)).dot(predicted_covariance)     # New Covariance matrix (ex-post)

    # If measure new state and observation noises
    if not rate is None:
        # Get state and observation error
        update_s = updated_state - state
        residual = y - x.dot(updated_state)

        # Update noise covariance matrices based on an EMA
        noise = (1 - rate) * noise + rate * update_s.dot(update_s.T)
        noise_obs = (1 - rate) * noise_obs + rate * residual.dot(residual.T)

        # Return the new state with the updated noise
        return updated_state, updated_covariance, noise, noise_obs

    # Return the new state
    return updated_state, updated_covariance


#######################################################################################################################

#######################################################################################################################
''' Test Function '''
#######################################################################################################################

#######################################################################################################################

def test_linear_regression():
    # Generate synthetic data
    np.random.seed(10)
    
    dim = 1
    n   = 10000  # Number of data points

    beta    = np.ones((1, dim+1, 1)) + 0.05*np.random.randn(n, dim+1, 1).cumsum(axis=0)
    x       = np.concatenate((np.random.randn(n, dim, 1), np.ones((n, 1, 1))), axis=1)
    y       = (x*beta).sum(axis=1).reshape(-1, 1, 1) + 0.1*np.random.randn(n, 1, 1)

    # Initialize Kalman filter
    state           = np.zeros((n+1, dim+1, 1))  # Initial state
    covariance      = np.zeros((n+1, dim+1, dim+1))  # Initial covariance
    covariance[0]   = np.eye(dim+1)

    noise       = 0
    noise_obs   = 0

    # Apply Kalman filter to each data point
    for i in range(n):
        state[i+1], covariance[i+1], noise, noise_obs   = kalman_step(x[i].T, y[i], state[i], covariance[i], noise, noise_obs, rate=1e-1)

    #dist    = np.linalg.norm(state.reshape(-1, dim+1)[1:] - beta.reshape(-1, dim+1), axis=1)

    fig, ax = pl.subplots(1,1)

    #ax.plot(dist, label="Distance with True Value")
    ax.plot(np.squeeze(state[:,0,:]), label="Kalman filter")
    ax.plot(np.squeeze(beta[:,0,:]), color="r", linestyle="--", label="True value")
    #ax.axhline(1, color="r", linestyle="--", label="True value")

    ax.grid()
    ax.legend()

    pl.show()

#######################################################################################################################

def test_moving_average():
    # Generate synthetic data
    np.random.seed(10)
    
    n   = 10000  # Number of data points

    mean    = 0.05*np.random.randn(n, 1, 1).cumsum(axis=0)
    y       = mean + np.random.randn(n, 1, 1)

    # Initialize Kalman filter
    state           = np.zeros((n+1, 1, 1))  # Initial state
    covariance      = np.zeros((n+1, 1, 1))  # Initial covariance
    covariance[0]   = np.eye(1)

    noise       = 0.05
    noise_obs   = 100

    # Apply Kalman filter to each data point
    for i in range(n):
        state[i+1], covariance[i+1] = kalman_step(np.ones((1,1)), y[i], state[i], covariance[i], noise, noise_obs, rate=None)

    fig, ax = pl.subplots(1,1)

    #ax.plot(dist, label="Distance with True Value")
    ax.plot(np.squeeze(y[:,0,:]), color="g", linestyle="--", label="Observed")
    ax.plot(np.squeeze(state[:,0,:]), label="Kalman filter")
    ax.plot(np.squeeze(mean[:,0,:]), color="r", linestyle="--", label="True value")
    #ax.axhline(1, color="r", linestyle="--", label="True value")

    ax.grid()
    ax.legend()

    pl.show()

#######################################################################################################################

def test_moving_average_and_linear_regression():
    # Generate synthetic data
    np.random.seed(10)
    
    n   = 10000  # Number of data points

    std             = 0.01
    mean_rev_rate   = 5e-3
    residual        = np.zeros((n, 1, 1))
    for k in range(n):
        residual[k] = std*np.sqrt(2*mean_rev_rate*(1-mean_rev_rate))*np.random.randn(1, 1) + (1 - mean_rev_rate)*residual[k-1]

    beta    = np.ones((1, 2, 1)) + 0.05*np.random.randn(n, 2, 1)    # .cumsum(axis=0)
    x       = np.concatenate((0.1*np.ones((n, 1, 1)), np.ones((n, 1, 1))), axis=1)
    y       = (x*beta).sum(axis=1).reshape(-1, 1, 1) + residual
    
    x_obs   = x[:,:1,:] + 0.1*np.random.randn(n, 1, 1)
    y_obs   = y + 0.1*np.random.randn(n, 1, 1)

    # Initialize Kalman filter
    beta_state      = np.zeros((n+1, 2, 1))  # Initial state
    covariance      = np.zeros((n+1, 2, 2))  # Initial covariance
    covariance[0]   = np.eye(2)

    beta_state_0    = np.zeros((n+1, 2, 1))  # Initial state
    covariance_0    = np.zeros((n+1, 2, 2))  # Initial covariance
    covariance_0[0] = np.eye(2)

    x_state     = np.zeros((n+1, 1, 1))  # Initial state
    x_cov       = np.zeros((n+1, 1, 1))  # Initial covariance
    x_cov[0]    = np.eye(1)

    y_state     = np.zeros((n+1, 1, 1))  # Initial state
    y_cov       = np.zeros((n+1, 1, 1))  # Initial covariance
    y_cov[0]    = np.eye(1)

    noise       = .01
    noise_obs   = 100

    # Apply Kalman filter to each data point
    for i in range(n):
        x_0 = np.array([x_obs[i,0,0], 1]).reshape(2, 1)

        beta_state_0[i+1], covariance_0[i+1] = kalman_step(x_0.T, y_obs[i], beta_state_0[i], covariance_0[i], noise, noise_obs, rate=None)

        x_state[i+1], x_cov[i+1] = kalman_step(np.ones((1,1)), x_obs[i], x_state[i], x_cov[i], noise, noise_obs, rate=None)
        y_state[i+1], y_cov[i+1] = kalman_step(np.ones((1,1)), y_obs[i], y_state[i], y_cov[i], noise, noise_obs, rate=None)

        x_0 = np.array([x_state[i+1, 0, 0], 1]).reshape(2, 1)

        beta_state[i+1], covariance[i+1]    = kalman_step(x_0.T, y_state[i+1], beta_state[i], covariance[i], 0.05, 1, rate=None)

    x_obs   = np.concatenate((x_obs, np.ones((n, 1, 1))), axis=1)
    x_state = np.concatenate((x_state, np.ones((n+1, 1, 1))), axis=1) 

    residual    = residual.reshape(-1)
    residual_0  = y_obs.reshape(-1) - (x_obs*beta_state_0[1:]).sum(axis=1).reshape(-1)
    residual_1  = y_state[1:].reshape(-1) - (x_state[1:]*beta_state[1:]).sum(axis=1).reshape(-1)

    fig, ax = pl.subplots(2,2)
    ax  = ax.flatten()

    ax[0].plot((residual_0 - residual)/(np.abs(residual_0) + np.abs(residual)), 'k', label="Residual 0")
    ax[0].plot((residual_1 - residual)/(np.abs(residual_1) + np.abs(residual)), 'r', label="Residual 1")

    #ax[1].plot(residual_0, 'k', label="Residual 0")
    ax[1].plot(residual_1, 'r', label="Residual 1")
    ax[1].plot(residual, 'b', label="Residual")

    ax[2].plot(np.squeeze(y_obs), 'k', label="Observed")
    ax[2].plot(np.squeeze(y_state), 'r', label="Kalman filter")
    ax[2].plot(np.squeeze(y), 'b', label="True value")

    ax[3].plot(np.squeeze(x_obs[:,0,:]), 'k', label="Observed")
    ax[3].plot(np.squeeze(x_state[:,0,:]), 'r', label="Kalman filter")
    ax[3].plot(np.squeeze(x[:,0,:]), 'b', label="True value")

    for a in ax:
        a.grid()

    pl.show()

#######################################################################################################################

def johansen_test():
    from scipy.linalg import block_diag

    # Generate synthetic data
    np.random.seed(10)
    
    dim = 2
    n   = 10000  # Number of data points
    
    Q       = np.random.randn(dim, dim)
    Q, _    = np.linalg.qr(Q)

    A   = - np.diag(np.abs(np.random.randn(dim)))

    A   = Q.dot(A.dot(Q.T))
    b   = np.random.randn(dim)

    y   = np.zeros((n+1, dim))
    for k in range(n):
        y[k+1]  = A.dot(y[k]) + b + np.random.randn(dim)
    
    # Initialize Kalman filter
    state           = np.zeros((n+1, dim*(dim+1), 1))           # Initial state
    covariance      = np.zeros((n+1, dim*(dim+1), dim*(dim+1))) # Initial covariance
    covariance[0]   = np.eye(dim*(dim+1))

    noise       = 1.
    noise_obs   = 1.

    fig = pl.figure()
    pl.plot(np.linalg.norm(y, axis=1))
    pl.grid()
    pl.show()

    for i in range(n):
        x   = [np.concatenate([y[i], np.ones(1)], axis=0)]*dim
        x   = block_diag(*x)
        print(x)
        print(y[i+1])

        state[i+1], covariance[i+1] = kalman_step(x, y[i+1].reshape(-1,1), state[i], covariance[i], noise, noise_obs, rate=None)

    state   = state.reshape(n+1, -1)
    A       = np.concatenate((A, b.reshape(-1,1)), axis=1).reshape(1, -1)
    dist    = np.linalg.norm(state[1:] - A, axis=1)
    print(dist)

    fig, ax = pl.subplots(1,1)

    ax.plot(dist, label="Distance with True Value")

    ax.grid()
    ax.legend()

    pl.show()







#######################################################################################################################

if __name__ == "__main__":
    import pylab    as pl

    #test_linear_regression()
    #test_moving_average()
    #test_moving_average_and_linear_regression()
    johansen_test()