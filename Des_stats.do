********************************************************************************
******************** Master Thesis: Descriptive Statistics *********************
********************************************************************************
** Institution: KU Leuven
** Program: Master of Economics
** Author: Valeria Córdova
** Date last change: April 23, 2021
** Data: National Survey on Drug Use and Health (NSDUH) 2018 and 2019, USA.

use "${final}\Final_sample.dta", clear
set matsize 2000

preserve
	qui oprobit alcuse_num alcfage smokeqnt mjnfq cocfq age sex race health mhealth married emplmnt educcat fincome 	socialsec hhkids relserv
	gen smpl = e(sample)

	keep if smpl == 1
	
	// Create matrix of descriptive statistics for continuous variables:
	matrix conts = J(1,5,.)
	mat colnames conts = "N" "mean" "sd" "min" "max" 
	
	** Fill first row: Overall
	qui sum alcfage
	mat conts[1,1] = r(N)
	mat conts[1,2] = r(mean)
	mat conts[1,3] = r(sd)
	mat conts[1,4] = r(min)
	mat conts[1,5] = r(max)
	
	local conts "mjnfq cocfq mhealth"	
	foreach var of varlist `conts' {
	    if "`var'" == "mhealth" {
		    mat aux_conts = J(1,5,.)
		    
			qui sum `var'
			mat aux_conts[1,1] = r(N)
			mat aux_conts[1,2] = r(mean)
			mat aux_conts[1,3] = r(sd)
			mat aux_conts[1,4] = r(min)
			mat aux_conts[1,5] = r(max)
			
			mat conts = (conts \ aux_conts)	
		}
		else {
		    mat aux_conts_0 = J(1,5,.)
		    mat aux_conts_1 = J(1,5,.)
			
			qui sum `var'
			mat aux_conts_0[1,1] = r(N)
			mat aux_conts_0[1,2] = r(mean)
			mat aux_conts_0[1,3] = r(sd)
			mat aux_conts_0[1,4] = r(min)
			mat aux_conts_0[1,5] = r(max)
			
			mat conts = (conts \ aux_conts_0)	
			
			qui sum `var' if `var' > 0
			mat aux_conts_1[1,1] = r(N)
			mat aux_conts_1[1,2] = r(mean)
			mat aux_conts_1[1,3] = r(sd)
			mat aux_conts_1[1,4] = r(min)
			mat aux_conts_1[1,5] = r(max)
			
			mat conts = (conts \ aux_conts_1)
		}
	}
	
	mat rownames conts = "alcfage"								    	 	///
						 "mjnfq_0"											///
						 "mjnfq_1"											///
						 "cocfq_0" 										 	///
						 "cocfq_1"											///
						 "mhealth"	 								
						 
	// Save matrix to Excel sheet
	putexcel set "${main}\Document\Output\Tables\Des_cont", sheet("Raw", replace) modify
	putexcel A1 = matrix(conts), nfor(#.00) names
	putexcel save
						 
	// Create matrix of descriptive statistics for categorical variables:
	// Age, sex, ethnicity, educational level, employment status, marital status, 
	// income, # kids, health, smokeqnt, survey year 
	matrix cats = J(1,5,.)
	mat colnames cats = "N" "prev.1" "prev.2" "prev.3" "prev.4" 
						 
	** Fill first row: Overall
	qui sum alcuse_num
	mat cats[1,1] = r(N)
	local N = r(N)

	levelsof alcuse_num, local(cat)
	foreach i in `cat' {
		qui sum alcuse_num if alcuse_num == `i'
		mat cats[1,`i'+2] = (r(N)/`N')*100
	}					 
		
	local vars "sex age race educcat emplmnt married fincome socialsec hhkids health smokeqnt relserv year"	
	foreach var of varlist `vars' {	
		levelsof `var', local(lvls)
		foreach lvl in `lvls' {
			mat aux_stats = J(1,5,.)
			qui sum `var' if `var' == `lvl'
			mat aux_stats[1,1] = r(N)
			local N_var = r(N)
			
			foreach i in `cat' {
				qui sum `var' if `var' == `lvl' & alcuse_num == `i'
				mat aux_stats[1,`i'+2] = (r(N)/`N_var')*100
			}
			mat cats = (cats \ aux_stats)		
		}
	}

	mat rownames cats = "Overall"											 	///
						"Sex.0" "Sex.1"										 	///
						"Age.0" "Age.1" "Age.2" "Age.3" "Age.4" "Age.5"		 	///
						"R.1" "R.2" "R.3" "R.4" "R.5" "R.6" 				 	///
						"Ed.1" "Ed.2" "Ed.3" "Ed.4"							 	///
						"Em.0" "Em.1" "Em.2" "Em.3" "Em.4" "Em.5" "Em.6"	 	///
						"Ma.1" "Ma.2" "Ma.3" "Ma.4"							 	///
						"Fi.1" "Fi.2" "Fi.3" "Fi.4" "Fi.5" "Fi.6" "Fi.7"	 	///
						"Ss.0" "Ss.1"											///
						"hhk.0" "hhk.1" "hhk.2" "hhk.3"						 	///
						"He.1" "He.2" "He.3" "He.4" "He.5"						///
						"Sm.0" "Sm.1" "Sm.2" "Sm.3" "Sm.4" "Sm.5" "Sm.6" "Sm.7" ///
						"Rs.1" "Rs.2" "Rs.3" "Rs.4" "Rs.5" "Rs.6"				/// 
						"y.2018" "y.2019"

	// Save matrix to Excel sheet
	putexcel set "${main}\Document\Output\Tables\Des_cat", sheet("Raw", replace) modify
	putexcel A1 = matrix(cats), nfor(#.00) names
	putexcel save

restore