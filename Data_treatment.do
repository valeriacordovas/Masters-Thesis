********************************************************************************
************************ Master Thesis: Data treatment *************************
********************************************************************************
** Institution: KU Leuven
** Program: Master of Economics
** Author: Valeria Córdova
** Date last change: April 01, 2021
** Data: National Survey on Drug Use and Health (NSDUH) 2018 and 2019, USA.

**** Preamble
/*
clear all 
set more off

	global main    "G:\Mi unidad\Master\Thesis"
	global raw 	   "${main}\Data\Raw\NSDUH"
	global treated "${main}\Data\Treated"
	global final   "${main}\Data\Final"
*/	

****** Initialize ******
foreach year in 2018 2019 {	
	**** Load data
	use "${raw}\NSDUH_`year'", clear


	**** Select and rename variables
	** ID variables: year QUESTID2
	** Dependent variables: iralcfm NODR30A/ALCUS30D
	** Independent variables: alctry AGE2 irsex IRPINC3 IRFAMIN3 NEWRACE2 IREDUHIGHST2 
	**						  EDUCCAT2 schenrl health irmarit (irmaritstat)  
	** 						  IRHHSIZ2 IRKI17_2 snrlgimp snrldcsn irfamsoc WRKSTATWK2
	** Drug consumption: ircigfm CIG30AV ircigrc irmjfm irmjrc ircocfm ircocrc
	/*
	keep year QUESTID2 iralcfm NODR30A alctry AGE2 irsex IRPINC3 IRFAMIN3 NEWRACE2 ///
		 IREDUHIGHST2 EDUCCAT2 schenrl health irmarit irmaritstat IRHHSIZ2 		   ///
		 IRKI17_2 snrlgimp snrldcsn irfamsoc ircigfm CIG30AV ircigrc irmjfm irmjrc ///
		 ircocfm ircocrc
	*/
	keep QUESTID2 ///
		 iralcfm II2ALCFM ALCUS30D iralcage iialcage	///
		 IRALCBNG30D IIALCBNG30D						///
		 AGE2 irsex NEWRACE2   							///
		 IREDUHIGHST2 IIEDUHIGHST2 eduhighcat eduschlgo ///
		 health K6SCMON									///
		 irmarit iimarit 								///
		 IRHHSIZ2 IIHHSIZ2 IRKI17_2 IIKI17_2	 		///
		 WRKSTATWK2										///
		 IRPINC3 IIPINC3 IRFAMIN3 IIFAMIN3 irfamsoc		///
		 snrlgimp snrldcsn snrlfrnd snrlgsvc			///
		 ircigfm iicigfm ircigrc iicigrc CIG30AV		///
		 irmjfm iimjfm irmjrc iimjrc					///
		 ircocfm iicocfm ircocrc iicocrc	 
		 
	rename QUESTID2 ID
	rename iralcfm alcfq
	rename II2ALCFM alcfq_IM
	rename ALCUS30D alcqnt
	rename iralcage alcfage
	rename iialcage alcfage_IM
	rename IRALCBNG30D bingedrk
	rename IIALCBNG30D bingedrk_IM
	rename AGE2 age0 
	rename irsex sex0
	rename NEWRACE2 race0
	rename IREDUHIGHST2 educ
	rename IIEDUHIGHST2 educ_IM
	rename eduhighcat educcat
	rename eduschlgo schenrl0
	rename K6SCMON mhealth
	rename health health0
	rename irmarit married
	rename iimarit married_IM
	rename IRHHSIZ2 hhsize
	rename IIHHSIZ2 hhsize_IM
	rename IRKI17_2 hhkids
	rename IIKI17_2 hhkids_IM
	rename WRKSTATWK2 emplmnt
	rename IRPINC3 rincome
	rename IIPINC3 rincome_IM
	rename IRFAMIN3 fincome
	rename IIFAMIN3 fincome_IM
	rename irfamsoc socialsec0
	rename snrlgimp imprel
	rename snrldcsn infrel
	rename snrlfrnd frnrel
	rename snrlgsvc relserv
	rename ircigfm smokefq
	rename iicigfm smokefq_IM
	rename ircigrc smokerec
	rename iicigrc smokerec_IM
	rename CIG30AV smokeqnt0
	rename irmjfm mjnfq
	rename iimjfm mjnfq_IM
	rename irmjrc mjnrec
	rename iimjrc mjnrec_IM
	rename ircocfm cocfq
	rename iicocfm cocfq_IM
	rename ircocrc cocrec
	rename iicocrc cocrec_IM


	**** Recode variables
	replace alcfq = . if alcfq_IM == 3 			//inlist(alcfq_IM,3,4)
	replace bingedrk = . if bingedrk_IM == 3 	//inlist(bingedrk_IM,3,4)
	local vars "alcfage educ married hhsize hhkids rincome fincome smokefq smokerec mjnfq mjnrec cocfq cocrec"
	foreach v of varlist `vars' {
		replace `v' = . if `v'_IM == 3
	}

	recode alcfq (93 = 0)
	recode bingedrk (93 = 0)
	recode alcqnt (993 = 0) (975 985 994 997 998 = .)
	recode age0 (1/9 = .)													///
				(10 11 12 = 0 "21 to 25 years old")							///
				(13       = 1 "26 to 29 years old")  						///
				(14       = 2 "30 to 34 years old")  						///
				(15       = 3 "35 to 49 years old")  						///
				(16       = 4 "50 to 64 years old")  						///
				(17       = 5 "65 years old or older"), gen(age)
	recode sex0 (1 = 0 "Male") (2 = 1 "Female"), gen(sex)
	recode health0 (94 97 = .) 												///
				   (1 = 5 "Excellent") 										///
				   (2 = 4 "Very good")										///
				   (3 = 3 "Good")											///
				   (4 = 2 "Fair") 											///
				   (5 = 1 "Poor"), gen(health)
	recode schenrl0 (2 = 0 "No") (1 = 1 "Yes") 		 						///
					(11 85 94 97 98 = .), gen(schenrl)
	rename hhkids hhkids0
	recode hhkids0 (1 = 0 "No children under 18")  							///
				   (2 = 1 "One child under 18")   							///
				   (3 = 2 "Two children under 18")							///
				   (4 = 3 "Three or more children under 18"), gen(hhkids)
	rename emplmnt emplmnt0
	recode emplmnt0 (1 2 3 = 0 "Employed")									///
					(4 = 1 "Unemployed/on layoff, looking for work")		///
					(5 = 2 "Disabled")										///
					(6 = 3 "Keeping house full-time")						///
					(7 = 4 "In school/training")							///
					(8 = 5 "Retired")										///
					(9 = 6 "Unemployed")									///
					(98 99 = .), gen(emplmnt)
	recode socialsec0 (2 = 0 "No") (1 = 1 "Yes"), gen(socialsec)
	recode smokefq (93 = 0)
	rename smokerec smokerec0
	recode smokerec0 (9 = 0 "Never smoked cigarettes")						///
					 (1 = 1 "30 days or less ago")							///
					 (2 = 2 "Between 31 days and 1 year ago")				///
					 (3 = 3 "Between 1 and 3 years ago")					///
					 (4 = 4 "More than 3 years ago"), gen(smokerec)
	recode smokeqnt0 (91 93 = 0 "Did not use cigarettes in the past") 			///
					 (1  = 1 "Less than 1 per day")							///
					 (2  = 2 "1 per day")									///
					 (3  = 3 "2 to 5 per day")								///
					 (4  = 4 "6 to 15 per day")								///
					 (5  = 5 "16 to 25 per day")							///
					 (6  = 6 "26 to 35 per day")							///
					 (7  = 7 "More than 35 per day")						///
					 (85 94 97 98 = .), gen(smokeqnt)
	recode mjnfq (93 = 0)
	rename mjnrec mjnrec0
	recode mjnrec0 (9 = 0 "Never used marijuana")							///
				   (1 = 1 "30 days or less ago")							///
				   (2 = 2 "Between 31 days and 1 year ago")					///
				   (3 = 3 "More than 1 year ago"), gen(mjnrec)
	recode cocfq (93 = 0)
	rename cocrec cocrec0
	recode cocrec0 (9 = 0 "Never used cocaine")								///
				   (1 = 1 "30 days or less ago")							///
				   (2 = 2 "Between 31 days and 1 year ago")					///
				   (3 = 3 "More than 1 year ago"), gen(cocrec)
	recode relserv (85 94 97 98 = .)
	recode imprel (85/99 = .)
	recode infrel (85/99 = .)
	recode frnrel (85/99 = .)
	recode race0 (1 = 1 "NonHisp White") 									///
				 (2 = 2 "NonHisp Afr Am")									///
				 (3 4 = 3 "NonHisp Native") 								///	
				 (5 = 4 "NonHisp Asian")									///
				 (6 = 5 "NonHisp more than one race")						///
				 (7 = 6 "Hispanic"), gen(race)
	
	** Generate dependent variable (categorical) 
	gen alcuse = ""
	
	// Low-risk use (Moderation - US dietary guidelines)
	replace alcuse = "2.Moderate" if (alcqnt <= 1 & sex == 1) | (alcqnt <= 2 & sex == 0)
	
	// No use
	replace alcuse = "1.No use" if alcfq == 0 | alcqnt == 0 
	
	// At-risk use (CDC)
	replace alcuse = "3.Misuse" if (alcqnt > 1 & sex == 1) | (alcqnt > 2 & sex == 0)
	
	// Binge drinking
	replace alcuse = "4.Binge" if bingedrk >= 1
	
	encode alcuse, gen(alcuse_num0)
	recode alcuse_num0 (1 = 0 "No use") 									///
					   (2 = 1 "Moderate")									///
					   (3 = 2 "Misuse")										///
					   (4 = 3 "Binge"), gen(alcuse_num)
	
	** Additional variables
	gen year = `year'
	gen cons = 1
	
	gen smkever = (smokefq != 91)
	replace smkever = . if smokefq == .

	gen mjnever = (mjnfq != 91)
	replace mjnever = . if mjnfq == .

	gen cocever = (cocfq != 91)
	replace cocever = . if cocfq == .
	
	replace smokefq = floor(smokefq)
	replace mjnfq = floor(mjnfq)
	replace cocfq = floor(cocfq)

	recode smokefq (91 = 0)
	recode mjnfq (91 = 0)
	recode cocfq (91 = 0)

	*gen religious = (imprel + infrel + frnrel)/3
	gen religious = infrel
	
	** Label variables and values
	label define dum 0 "No" 1 "Yes"
	label values smkever dum
	label values mjnever dum
	label values cocever dum

	label variable ID "Respondent ID"
	label variable alcuse "Alcohol use categories"
	label variable alcuse_num "Alcohol use categories (numeric)"
	label variable alcqnt "Usual # of drinks per day past 30 days"
	label variable bingedrk "Binge alcohol frequency past 30 days"
	label variable smokefq "Cigarette use frequency past 30 days"
	label variable alcfq "Alcohol use frequency past 30 days"
	label variable mjnfq "Marijuana use frequency past 30 days"
	label variable cocfq "Cocaine use frequency past 30 days"
	label variable alcfage "Age of first alcohol use"
	label variable relserv "# of religious services attended in the past year"
	label variable imprel "My religious beliefs are very important"
	label variable infrel "My religious beliefs influence my decisions"
	label variable frnrel "It is important that my friends share religious beliefs"
	label variable health "Overall health"
	label variable mhealth "K6 mental health total score past month"
	label variable married "Marital status"
	label variable educ "Educaction"
	label variable race "Race"
	label variable educcat "Education categories"
	label variable hhsize "Household size"
	label variable emplmnt "Employment status"
	label variable rincome "Respondent's total income"
	label variable fincome "Total family income"
	label variable age "Age"
	label variable sex "Sex"
	label variable schenrl "Now enrolled in educational institution"
	label variable hhkids "# children < 18 in household"
	label variable socialsec "Family receives SS or RR payments"
	label variable smokerec "Cigarette recency"
	label variable smokeqnt "Average # cigarettes smoked per day on days smoked"
	label variable mjnrec "Marijuana recency"
	label variable cocrec "Cocaine recency"
	label variable smkever "Ever smoked cigarettes"
	label variable mjnever "Ever used marijuana"
	label variable cocever "Ever used cocaine"
	label variable religious "Religiousness index: 1-Not religious 4-Very religious"



	**** Select sample
	drop if age == .
	drop if inlist(alcfq,.,91) | inlist(alcqnt,.,991) | inlist(bingedrk,.,91)
	drop if alcfq != floor(alcfq)
	
	
	**** Save data base
	drop ID *_IM *0
	gen ID = _n
	
	order ID year cons alcuse alcuse_num alcfq alcqnt bingedrk alcfage 			///
		  smkever smokefq smokerec smokeqnt	mjnever mjnfq mjnrec 				///
		  cocever cocfq cocrec			 										///
		  age sex race health mhealth married emplmnt educ educcat schenrl		///
		  rincome fincome socialsec hhsize hhkids 								///
		  relserv religious imprel infrel frnrel
			
	save "${treated}\NSDUH_`year'_mod.dta", replace

}


**** Append databases
use "${treated}\NSDUH_2018_mod.dta", clear

append using "${treated}\NSDUH_2019_mod.dta"

gen yearbin = (year == 2019)

save "${treated}\NSDUH_1819.dta", replace


**** Select final sample
// 62,986 obs: 31,797 (2018) + 31,189 (2019)
use "${treated}\NSDUH_1819.dta", clear

*export delimited using "${final}\Final_sample.csv", nolabel replace
save "${final}\Final_sample.dta", replace
export delimited using "${final}\Final_sample.csv", replace

/*
preserve
	drop alcqnt
	
	export delimited using "${final}\Fq_sample.csv", nolabel replace
	save "${final}\Fq_sample.dta", replace

restore

preserve
	drop alcfq
	
	export delimited using "${final}\Qnt_sample.csv", nolabel replace
	save "${final}\Qnt_sample.dta", replace

restore

/*
gen n = 30
 glm alcfq alcfage i.age i.smkever i.mjnever i.sex i.race i.educ i.fincome, family(binomial n)
poisson alcfq alcfage i.age i.smkever i.mjnever i.sex i.race i.educ i.fincome


