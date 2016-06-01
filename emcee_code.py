import numpy as np, emcee, os, corner

#FUNCTIONS

def modelo(propiedades, parametros):

    num_filas = len(propiedades[:,0])
    num_parametros = len(parametros)

    resultado = np.zeros(num_filas)

    for i in range(num_filas):

    	suma = 0

    	for j in range(num_parametros):

    		coeficiente = parametros[j]
    		propiedad = propiedades[i,j]

        	suma += coeficiente*propiedad

        resultado[i] = suma

    return resultado


def lnprob(parametros, propiedades, mic):

    #coeficientes
    for par in parametros:
    	if(par < -100.0 or par > 100.0):
	    	return -np.inf

    code = modelo(propiedades, parametros)

    chi_squared = np.sum((code-mic)**2)

    return -0.5*chi_squared

#emcee
def emcee_code_function(propiedades, mic, semilla):

    #Semilla
    np.random.seed(semilla)

    #Valores iniciales
    num_parametros = len(propiedades[0,:])
    parametros = np.random.uniform(-10,10, num_parametros)

    #Running emcee
    ndim = num_parametros
    nwalkers = num_parametros*2
    nsteps = 4000*num_parametros

    pos = [parametros+ 1e-3*np.random.randn(ndim) for i in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(propiedades, mic), threads=8)

    sampler.run_mcmc(pos, nsteps)

    # Saving results
    samples_fc = sampler.flatchain
    logprob_fc = sampler.flatlnprobability

    np.savetxt('sampler_flatchain.dat', samples_fc, delimiter=',')

    #This number should be between approximately 0.25 and 0.5 if everything went as planned.
    print("Mean acceptance fraction: {0:.3f} (Should be between 0.25 and 0.5 approximately)".format(np.mean(sampler.acceptance_fraction)))

    #Discard the initial 50 steps
    samples = samples_fc[50:]
    logprob = logprob_fc[50:]

    # Unpack the walk for each parameter
    parametros_caminata = np.transpose(samples)
    parametros_emcee = np.zeros((num_parametros, 3))

    # Extract the percentiles for each parameter
    for i in range(num_parametros):
        fila = parametros_caminata[i,:]
        parametros_emcee[i] = np.percentile(fila, [16, 50, 84])

    labels_alfas = []

    # Prints them
    print('Parameter = [16 50 84]')

    for i in range(num_parametros):
        alfa_i = 'alfa'+str(i+1)
        labels_alfas.append(alfa_i)
        print(alfa_i+' = ', parametros_emcee[i])


    fig = corner.corner(samples, labels = labels_alfas, quantiles = [0.16, 0.5, 0.84])
    fig.savefig("triangle.png",dpi=200)

    chi2 = np.zeros(nsteps)

    for i in range(nsteps):
        params = samples_fc[i,:]
        chi2[i] = -2.0*lnprob(params, propiedades, mic)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(chi2)
    plt.xlabel('step')
    plt.ylabel('$\chi^2$')
    plt.savefig('chi_squared.png')

    plt.figure()
    plt.plot(np.log(chi2))
    plt.xlabel('step')
    plt.ylabel('$\log{\chi^2}$')
    plt.savefig('chi_squared_log.png')

    return parametros_emcee
