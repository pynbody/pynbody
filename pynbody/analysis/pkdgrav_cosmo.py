"""

pkdgrav_cosmo
=============

Cosmological module from PKDGRAV.

N.B.  This code is being shared with skid and the I.C. generator.

**NEEDS DOCUMENTATION**
"""

import math
from scipy.integrate import romberg, ode


class Cosmology:
    """ docs placeholder """

    EPSCOSMO = 1e-7

    def __init__(self, sim=None, H0=math.sqrt(8*math.pi/3), Om=0.272,
                 L=0.728, Ob=0.0456,
                 Or=0.0, Quin=0.0, Ok=0.0):
        if sim is not None:
            self.dOmegaM = sim.properties['omegaM0']
            self.dLambda = sim.properties['omegaL0']
        else:
            self.dOmegaM = Om
            self.dLambda = L
        self.dHubble0 = H0
        self.dOmegab = Ob
        self.dOmegaRad = Or
        self.dQuintess = Quin
        self.dOmegaCurve = Ok
        self.bComove = 1

    # The cosmological equation of state is entirely determined here.  We
    # will derive all other quantities from these two functions.
    def Exp2Hub(self, dExp):
        dOmegaCurve = (1.0 - self.dOmegaM - self.dLambda -
                       self.dOmegaRad - self.dQuintess)

        assert(dExp > 0.0)
        return (self.dHubble0*math.sqrt(self.dOmegaM*dExp
                                        + dOmegaCurve*dExp*dExp
                                        + self.dOmegaRad
                                        + self.dQuintess*dExp *
                                        dExp*math.sqrt(dExp)
                                        + self.dLambda*dExp*dExp*dExp*dExp)/(dExp*dExp))

    # Return a double dot over a.
    def ExpDot2(self, dExp):
        return (self.dHubble0*self.dHubble0 *
                (self.dLambda - 0.5*self.dOmegaM/(dExp*dExp*dExp)
                 + 0.25*self.dQuintess/(dExp*math.sqrt(dExp))
                 - self.dOmegaRad/(dExp*dExp*dExp*dExp)))

    def Time2Hub(self, dTime):
        a = self.Time2Exp(dTime)
        assert(a > 0.0)
        return self.Exp2Hub(a)

    def CosmoTint(self, dY):
        dExp = dY**(2.0/3.0)
        assert(dExp > 0.0)
        return 2.0/(3.0*dY*self.Exp2Hub(dExp))

    def Exp2Time(self, dExp):
        dOmegaM = self.dOmegaM
        dHubble0 = self.dHubble0

        if(self.dLambda == 0.0 and self.dOmegaRad == 0.0 and
           self.dQuintess == 0.0):
            if (dOmegaM == 1.0):
                assert(dHubble0 > 0.0)
                if (dExp == 0.0):
                    return(0.0)
                return(2.0/(3.0*dHubble0)*dExp**1.5)

            elif (dOmegaM > 1.0):
                assert(dHubble0 >= 0.0)
                if (dHubble0 == 0.0):
                    B = 1.0/math.sqrt(dOmegaM)
                    eta = acos(1.0-dExp)
                    return(B*(eta-sin(eta)))

                if (dExp == 0.0):
                    return(0.0)

                a0 = 1.0/dHubble0/math.sqrt(dOmegaM-1.0)
                A = 0.5*dOmegaM/(dOmegaM-1.0)
                B = A*a0
                eta = acos(1.0-dExp/A)
                return(B*(eta-sin(eta)))

            elif (dOmegaM > 0.0):
                assert(dHubble0 > 0.0)
                if (dExp == 0.0):
                    return(0.0)
                a0 = 1.0/dHubble0/math.sqrt(1.0-dOmegaM)
                A = 0.5*dOmegaM/(1.0-dOmegaM)
                B = A*a0
                eta = acosh(dExp/A+1.0)
                return(B*(sinh(eta)-eta))

            elif (dOmegaM == 0.0):
                assert(dHubble0 > 0.0)
                if (dExp == 0.0):
                    return(0.0)
                return(dExp/dHubble0)

            else:
                #* Bad value.
                assert(0)
                return(0.0)

        else:
            # Set accuracy to 0.01 EPSCOSMO to make Romberg integration
            # more accurate than Newton's method criterion in Time2Exp. --JPG
            return romberg(self.CosmoTint, self.EPSCOSMO, dExp**1.5, tol=0.01*self.EPSCOSMO)

    def Time2Exp(self, dTime):
        dHubble0 = self.dHubble0

        dExpOld = 0.0
        dExpNew = dTime*dHubble0
        dDeltaOld = dExpNew
        # old change in interval
        dUpper = 1.0e38
        # bounds on root
        dLower = 0.0

        it = 0

            # Root find with Newton's method.
        while (math.fabs(dExpNew - dExpOld)/dExpNew > self.EPSCOSMO):
            f = dTime - self.Exp2Time(dExpNew)
            fprime = 1.0/(dExpNew*self.Exp2Hub(dExpNew))

            if(f*fprime > 0):
                dLower = dExpNew
            else:
                dUpper = dExpNew

            dExpOld = dExpNew
            dDeltaOld = f/fprime
            dExpNext = dExpNew + dDeltaOld
            # check if bracketed
            if((dExpNext > dLower) and (dExpNext < dUpper)):
                dExpNew = dExpNext
            else:
                dExpNew = 0.5*(dUpper + dLower)
            it += 1
            assert(it < 40)

        return dExpNew

    def ComoveDriftInt(self, dIExp):
        return -dIExp/(Exp2Hub(1.0/dIExp))

#* Make the substitution y = 1/a to integrate da/(a^2*H(a))
    def ComoveKickInt(self, dIExp):
        return -1.0/(self.Exp2Hub(1.0/dIExp))

#* This function integrates the time dependence of the "drift"-Hamiltonian.
    def ComoveDriftFac(self, dTime, dDelta):
        dOmegaM = self.dOmegaM
        dHubble0 = self.dHubble0

        if(self.dLambda == 0.0 and self.dOmegaRad == 0.0 and
           self.dQuintess == 0.0):
            a1 = self.Time2Exp(dTime)
            a2 = self.Time2Exp(dTime+dDelta)
            if (dOmegaM == 1.0):
                return((2.0/dHubble0)*(1.0/math.sqrt(a1) - 1.0/math.sqrt(a2)))

            elif (dOmegaM > 1.0):
                assert(dHubble0 >= 0.0)
                if (dHubble0 == 0.0):
                    A = 1.0
                    B = 1.0/math.sqrt(dOmegaM)

                else:
                    a0 = 1.0/dHubble0/math.sqrt(dOmegaM-1.0)
                    A = 0.5*dOmegaM/(dOmegaM-1.0)
                    B = A*a0

                    eta1 = acos(1.0-a1/A)
                    eta2 = acos(1.0-a2/A)
                    return(B/A/A*(1.0/tan(0.5*eta1) - 1.0/tan(0.5*eta2)))

            elif (dOmegaM > 0.0):
                assert(dHubble0 > 0.0)
                a0 = 1.0/dHubble0/math.sqrt(1.0-dOmegaM)
                A = 0.5*dOmegaM/(1.0-dOmegaM)
                B = A*a0
                eta1 = acosh(a1/A+1.0)
                eta2 = acosh(a2/A+1.0)
                return(B/A/A*(1.0/tanh(0.5*eta1) - 1.0/tanh(0.5*eta2)))

            elif (dOmegaM == 0.0):
                # YOU figure this one out!
                assert(0)
                return(0.0)

            else:
                # Bad value?
                assert(0)
                return(0.0)
        else:
            return romberg(self.ComoveDriftInt, 1.0/self.Time2Exp(dTime),
                           1.0/self.Time2Exp(dTime + dDelta), tol=self.EPSCOSMO)

    # This function integrates the time dependence of the "kick"-Hamiltonian.
    def ComoveKickFac(self, dTime, dDelta):
        dOmegaM = self.dOmegaM
        dHubble0 = self.dHubble0

        if (not self.bComove):
            return(dDelta)
        elif(self.dLambda == 0.0 and self.dOmegaRad == 0.0
                and self.dQuintess == 0.0):
            a1 = self.Time2Exp(dTime)
            a2 = self.Time2Exp(dTime+dDelta)
            if (dOmegaM == 1.0):
                return((2.0/dHubble0)*(math.sqrt(a2) - math.sqrt(a1)))

            elif (dOmegaM > 1.0):
                assert(dHubble0 >= 0.0)
                if (dHubble0 == 0.0):
                    A = 1.0
                    B = 1.0/math.sqrt(dOmegaM)

                else:
                    a0 = 1.0/dHubble0/math.sqrt(dOmegaM-1.0)
                    A = 0.5*dOmegaM/(dOmegaM-1.0)
                    B = A*a0

                eta1 = acos(1.0-a1/A)
                eta2 = acos(1.0-a2/A)
                return(B/A*(eta2 - eta1))

            elif (dOmegaM > 0.0):
                assert(dHubble0 > 0.0)
                a0 = 1.0/dHubble0/math.sqrt(1.0-dOmegaM)
                A = 0.5*dOmegaM/(1.0-dOmegaM)
                B = A*a0;
                eta1 = acosh(a1/A+1.0);
                eta2 = acosh(a2/A+1.0);
                return(B/A*(eta2 - eta1));

            elif (dOmegaM == 0.0):
                #* YOU figure this one out!
                assert(0);
                return(0.0);

            else:
                #* Bad value?
                assert(0);
                return(0.0);

        else:
            return romberg(self.ComoveKickInt, 1.0/self.Time2Exp(dTime),
                           1.0/self.Time2Exp(dTime + dDelta), tol=self.EPSCOSMO)

    def ComoveLookbackTime2Exp(self, dComoveTime):
        if (not self.bComove):
            return(1.0);
        else:
            dExpOld = 0.0;
            dT0 = self.Exp2Time(1.0);
            dTime = dT0 - dComoveTime;
            dExpNew;
            it = 0;

            if(dTime < self.EPSCOSMO):
                dTime = self.EPSCOSMO;
            dExpNew = self.Time2Exp(dTime);
            # Root find with Newton's method.
            while (fabs(dExpNew - dExpOld)/dExpNew > self.EPSCOSMO):
                dTimeNew = self.Exp2Time(dExpNew);
                f = (dComoveTime - self.ComoveKickFac(
                    dTimeNew, dT0 - dTimeNew))
                fprime = -1.0/(dExpNew*dExpNew*self.Exp2Hub(dExpNew));
                dExpOld = dExpNew;
                dExpNew += f/fprime;
                it += 1
                assert(it < 20);

        return dExpNew;

# delta[1] => deltadot
    def GrowthFacDeriv(self, dlnExp, dlnDelta, dlnDeltadot):
        dExp = exp(dlnExp);
        dHubble = self.Exp2Hub(dExp);

        dlnDeltadot[0] = dlnDelta[1];
        dlnDeltadot[1] = (-dlnDelta[1]*dlnDelta[1]
                          - dlnDelta[1]*(1.0 + self.ExpDot2(
                              dExp)/(dHubble*dHubble))
                           + 1.5*self.Exp2Om(dExp));

    def GrowthFac(self, dExp):
        dlnExpStart = -15;
        nSteps = 200;
        dlnExp = math.log(dExp);

        assert(dlnExp > dlnExpStart);

        dDStart[0] = dlnExpStart;
        dDStart[1] = 1.0;  # Growing mode

        integrator = ode(GrowthFacDeriv).set_integrator('dopri5')
#        , 2,
#        , dDEnd,  nSteps);
        integrator = integrator.set_initial_value(
            2, dlnExpStart, dDStart, dlnExp)
        dDEnd = integrator.integrate(t1, step=0, relax=0)
        flag = integrator.successful()
        return exp(dDEnd[0]);

    def GrowthFacDot(self, dExp):
        dlnExpStart = -15;
        nSteps = 200;
        dlnExp = math.log(dExp);
        dDStart[2];
        dDEnd[2];

        assert(dlnExp > dlnExpStart);

        dDStart[0] = dlnExpStart;
        dDStart[1] = 1.0;  # Growing mode

        integrator = ode(GrowthFacDeriv).set_integrator('dopri5')
#        , 2, dlnExpStart, dDStart, dlnExp, dDEnd,nSteps);
        return dDEnd[1]*self.Exp2Hub(dExp)*exp(dDEnd[0]);

# expansion dependence of Omega_matter
    def Exp2Om(self, dExp):
        dHubble = self.Exp2Hub(dExp);
        return (self.dOmegaM*self.dHubble0*self.dHubble0 /
                (dExp*dExp*dExp*dHubble*dHubble))
