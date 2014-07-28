for (int i=0; i<n; i++) {
    double xi = X1(i);
    double yi = Y1(i);
    double zi = Z1(i);
    /* find x indices */
    int x_top_ind = n_x_vals-1;
    int x_bot_ind = 0;
    while(x_top_ind > x_bot_ind+1) {
        int mid_ind = floor((x_top_ind-x_bot_ind)/2)+x_bot_ind;
        if (xi > X_VALS1(mid_ind)) x_bot_ind = mid_ind;
        else x_top_ind = mid_ind;
	}

    /* find y indices */
    int y_top_ind = n_y_vals-1;
    int y_bot_ind = 0;
    
    while(y_top_ind > y_bot_ind+1) {
        int mid_ind = floor((y_top_ind-y_bot_ind)/2)+y_bot_ind;
        if (yi > Y_VALS1(mid_ind)) y_bot_ind = mid_ind;
        else y_top_ind = mid_ind;
	}

    /* find z indices */
    int z_top_ind = n_z_vals-1;
    int z_bot_ind = 0;
    
    while(z_top_ind > z_bot_ind+1) {
        int mid_ind = floor((z_top_ind-z_bot_ind)/2)+z_bot_ind;
        if (zi > Z_VALS1(mid_ind)) z_bot_ind = mid_ind;
        else z_top_ind = mid_ind;
	}

    double z_fac = (zi - Z_VALS1(z_bot_ind)) / 
	(Z_VALS1(z_top_ind) - Z_VALS1(z_bot_ind));
    double y_fac = (yi - Y_VALS1(y_bot_ind)) / 
	(Y_VALS1(y_top_ind) - Y_VALS1(y_bot_ind));
    double x_fac = (xi - X_VALS1(x_bot_ind)) / 
	(X_VALS1(x_top_ind) - X_VALS1(x_bot_ind));
    double v0, v1, v00, v01, v10, v11, v000, v001, v010, v011;
    double v100, v101, v110, v111;
    /*for (v111 = VALS3(x_top_ind,y_top_ind,z_top_ind); isnan(v111); y_top_ind++) {
      }*/
    v111 = VALS3(x_top_ind,y_top_ind,z_top_ind);
    v110 = VALS3(x_top_ind,y_top_ind,z_bot_ind);
    v11 = z_fac*(v111 - v110) + v110;
    v011 = VALS3(x_bot_ind,y_top_ind,z_top_ind);
    v010 = VALS3(x_bot_ind,y_top_ind,z_bot_ind);
    v01 = z_fac*(v011 - v010) + v010;
    v101 = VALS3(x_top_ind,y_bot_ind,z_top_ind);
    v100 = VALS3(x_top_ind,y_bot_ind,z_bot_ind);
    v10 = z_fac*(v101 - v100) + v100;
    v001 = VALS3(x_bot_ind,y_bot_ind,z_top_ind);
    v000 = VALS3(x_bot_ind,y_bot_ind,z_bot_ind);
    v00 = z_fac*(v001 - v000) + v000;
    v1 = x_fac*(v11 - v10) + v10;
    v0 = x_fac*(v01 - v00) + v00;
    RESULT_ARRAY1(i) = y_fac*(v1-v0) + v0;
    
    }
