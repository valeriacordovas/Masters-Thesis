********************************************************************************
************************** Master Thesis: Main Script **************************
********************************************************************************
** Institution: KU Leuven
** Program: Master of Economics
** Author: Valeria Córdova
** Date: Feb - 2021
** Data: National Survey on Drug Use and Health (NSDUH) 2018 and 2019, USA.

clear all 
set more off

	global main    "G:\Mi unidad\Master\Thesis"
	global raw 	   "${main}\Data\Raw\NSDUH"
	global treated "${main}\Data\Treated"
	global final   "${main}\Data\Final"
	
	
** Data treatment
do "${main}\Code\Data_treatment"

** Descriptive statistics
do "${main}\Code\Des_stats"