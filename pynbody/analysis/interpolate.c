for (int i=0; i<n_stars; i++) {
    //if ((i % 10000) == 0) printf("i: %d\n",i);
    double meti = METALS1(i);
    double agei = AGE_STAR1(i);
    /* find metal indices first */
    int met_top_ind = n_met_grid-1;
    int met_bot_ind = 0;
    while(met_top_ind > met_bot_ind+1) {
        int mid_ind = floor((met_top_ind-met_bot_ind)/2)+met_bot_ind;
        if (meti > MET_GRID1(mid_ind)) met_bot_ind = mid_ind;
        else met_top_ind = mid_ind;
	}

    /* find age indices next */
    int age_top_ind = n_age_grid-1;
    int age_bot_ind = 0;
    
    while(age_top_ind > age_bot_ind+1) {
        int mid_ind = floor((age_top_ind-age_bot_ind)/2)+age_bot_ind;
        if (agei > AGE_GRID1(mid_ind)) age_bot_ind = mid_ind;
        else age_top_ind = mid_ind;
	}

    double age_fac = (agei - AGE_GRID1(age_bot_ind)) / 
	(AGE_GRID1(age_top_ind) - AGE_GRID1(age_bot_ind));
    double met_fac = (meti - MET_GRID1(met_bot_ind)) / 
	(MET_GRID1(met_top_ind) - MET_GRID1(met_bot_ind));
    double m11 = MAG_GRID2(met_top_ind,age_top_ind);
    double m01 = MAG_GRID2(met_bot_ind,age_top_ind);
    double m10 = MAG_GRID2(met_top_ind,age_bot_ind);
    double m00 = MAG_GRID2(met_bot_ind,age_bot_ind);
    double m1 = met_fac*(m11 - m10) + m10;
    double m0 = met_fac*(m01 - m00) + m00;
    /*	printf("AGB fMass:  %g; i: %d; j: %d; t1: %g t0: %g; \nAGB massFac: %g; metFac: %g\n",
		fMass,i,j,t1,t0,massFac,metFac);*/
    OUTPUT_MAGS1(i) = age_fac*(m1-m0) + m0;
    
    }
