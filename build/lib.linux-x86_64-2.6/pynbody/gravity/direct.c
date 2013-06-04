for (int pi=0; pi<nips; pi++) {
    M_BY_R1(pi) = 0;
    for (int j=0; j < 3; j++) M_BY_R22(pi,j) = 0;
    for (int i=0; i<n; i++) {
	//if ((i % 10000) == 0) printf("i: %d\n",i);
	double mass = MASS1(i);
	double dx = IPOS2(pi,0) - POS2(i,0);
	double dy = IPOS2(pi,1) - POS2(i,1);
	double dz = IPOS2(pi,2) - POS2(i,2);
	double drsoft = 1./sqrt(dx*dx + dy*dy + dz*dz + epssq);
        double drsoft3 = drsoft*drsoft*drsoft;
	M_BY_R1(pi) += mass * drsoft;
	M_BY_R22(pi,0) += mass * drsoft3 * dx;
	M_BY_R22(pi,1) += mass * drsoft3 * dy;
	M_BY_R22(pi,2) += mass * drsoft3 * dz;
	}
    }
