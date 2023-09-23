

def messange_maker(es,messages,type,msg):
	if es:
		if type == 'Warn':
			es.JournalWarn(msg)
		elif type == 'PrintOut':
			es.JournalOut(msg)
		return messages
	else:
		messages.append({'type': type, 'message': msg})
		return messages

def convert_to_dict_Equivalent_signal(Equivalent_signal):
	# 0		   1				2			3				4						5
	# Equivalent_signal		[range][damage of block][max of block][repetition][percent damage of block][average_mean_for_block]
	Equivalent_signal_corrected = []
	for rf in Equivalent_signal:
		Equivalent_signal_corrected.append({
			'range': rf[0],
			'damage_of_block': rf[1],
			'max_of_block': rf[2],
			'repetition': rf[3],
			'percent_damage_of_block': rf[4],
			'average_mean_for_block': rf[5]})
	return Equivalent_signal_corrected




def DamageEquivalentSignal(
		RainFlow_corrected, # Rainflow dictionary
		BlocksNumber, # Number of blocks for the proposed test
		MinimumNumberOfCycles, # Minimum nuber of cycles in the proposed signal
		TestProgRepetitions, # Repetitions of the entire test program
		slope, # Slope for the comparison
		Include_increased_block_damage, # It's complicated for now use 1
		es, # nCode object, for use outside of nCode just put here None
		):



	total_damage = 0
	MaxValue = 0
	for rf in RainFlow_corrected:
		total_damage = total_damage + rf['damage_of_cycle']
		if rf['range'] > MaxValue:
			MaxValue = rf['range']




	messages = []
	RainFlow = []
	for rf in RainFlow_corrected:
		RainFlow.append([rf['range'],
						rf['damage_of_cycle'],
						rf['cumul_damage'],
						rf['number'], # order number
						rf['percentage_cumulative_damage'],
						rf['max_of_cycle'],
						rf['cycle_repetitions'], 
						rf['min_of_cycle']]
						)




	#es.JournalOut('########################################')
	#es.JournalOut('########################################')
	#es.JournalOut(' ###############  Max Square Method')
	#es.JournalOut('########################################')
	#es.JournalOut('########################################')
	
	Total_area=0
	for a in RainFlow:
		Total_area=Total_area+a[0]*a[1]

	### Deep copy of RainFlow
	NewRainFlow=[]
	for a in RainFlow:
		NewRainFlow.append( list(a) )


	Blocks=[0.,total_damage]
	BlocksIndexed=[0,len(RainFlow)-1]
	for iz in range(BlocksNumber-1):
		TotalMaxSquare=0
		TotalDamageOfBlocks=0			
		for x in range(len(BlocksIndexed)-1):
			LowerBlockBorder=Blocks[x]
			UpperBlockBorder=Blocks[x+1]
			LowerBlockBorderIn=BlocksIndexed[x]
			UpperBlockBorderIn=BlocksIndexed[x+1]
			DamageOfBlock=0
            
			#UpperBlockBorder=Decimal(   str(UpperBlockBorder)[0:-1]+str(int(str(UpperBlockBorder)[-1])+2))
			#es.JournalOut('UpperBlockBorder '+str(   UpperBlockBorder    ))
			for a in range(len(NewRainFlow)):
				LastCumulDamage=NewRainFlow[a][2]
				
#					if NewRainFlow[a][2] >	 LowerBlockBorder and 	NewRainFlow[a][2] <=	 UpperBlockBorder:
				if a >	 LowerBlockBorderIn and 	a <=	 UpperBlockBorderIn:					
					DamageOfBlock=DamageOfBlock+NewRainFlow[a][1]
					MaxSquare=DamageOfBlock*(MaxValue-NewRainFlow[a][0])
					
					if MaxSquare >= TotalMaxSquare:
						TotalMaxSquare=MaxSquare
						DamageOfDivision=DamageOfBlock+LowerBlockBorder
						IndexOfDivision=a
						RangeOfDivision=NewRainFlow[a][0]
						SquareHight=MaxValue-NewRainFlow[a][0]
						LowerBoundForAdding=LowerBlockBorder
						LowerBoundForAddingIn=LowerBlockBorderIn
			TotalDamageOfBlocks=TotalDamageOfBlocks+DamageOfBlock
			#es.JournalOut('LastCumulDamage '+str(   LastCumulDamage    ))
			#es.JournalOut('Lower / Upper bound '+str(   LowerBlockBorder*100./total_damage    )+'  '+str(   UpperBlockBorder*100./total_damage    ))	
			#es.JournalOut('Percentage of division '+str(   DamageOfDivision*100./total_damage    ))
		#es.JournalOut('TotalDamageOfBlocks '+str(   TotalDamageOfBlocks/total_damage    ))
		#es.JournalOut('Blocks :'+str(   Blocks    ))				
		for ab in range(len(NewRainFlow)):
			#if NewRainFlow[ab][2] > LowerBoundForAdding and NewRainFlow[ab][2] <= DamageOfDivision:
			if ab > LowerBoundForAddingIn and ab <= IndexOfDivision:
				NewRainFlow[ab][0]=NewRainFlow[ab][0]+SquareHight

		Blocks.append(DamageOfDivision)
		BlocksIndexed.append(IndexOfDivision)
		BlocksIndexed.sort()
		#es.JournalOut('### Damage division added: ' +str(DamageOfDivision*100./total_damage)  )
		#es.JournalOut('### Index added: ' +str(IndexOfDivision)  )
		Blocks.sort()

	BlocksPercent=[]
	for b in Blocks:
		BlocksPercent.append(b*100./total_damage)



	#es.JournalOut('Square heigth'+str(   SquareHight    ))	
	#es.JournalOut('Max value'+str(   MaxValue    ))	
	#es.JournalOut('Blocks :'+str(   Blocks    ))	
	#es.JournalOut('Blocks (percentage) :'+str(   BlocksPercent    ))	
	#es.JournalOut(str(   DamageOfDivision*100./total_damage   ))
	#es.JournalOut('Range of division :'+str(   RangeOfDivision    ))	
	#es.JournalOut(str(   DivisionNum    ))		


	### Calculating number of cycles in the input signal
	TotalNumberOfCyclesInOriginalSig=0
	for r in RainFlow:
		TotalNumberOfCyclesInOriginalSig=TotalNumberOfCyclesInOriginalSig+r[6]
	TotalNumberOfCyclesInOriginalSig=TotalNumberOfCyclesInOriginalSig*TestProgRepetitions
	messages = messange_maker(es, messages, 'PrintOut',
							  'Total number of cycles in original signal: ' +str(TotalNumberOfCyclesInOriginalSig)
							  )
	#es.JournalOut('Total number of cycles in original signal: ' +str(TotalNumberOfCyclesInOriginalSig)  )
	
	##################################
	#  Defining equivalent signal
	#####################################
	
	##				0		1					2			3			4							5				6				7
	### RainFlow [range] [damage_of_cycle][cumul_damage][number][percentage_cumulative_damage][max_of_cycle][cycle_repetitions][min_of_cycle]
	
	#		    			   0		   1				2			3				4						5
	#Equivalent_signal		[range][damage of block][max of block][repetition][percent damage of block][average_mean_for_block]
	
	if TotalNumberOfCyclesInOriginalSig<MinimumNumberOfCycles:
		messages = messange_maker(es, messages, 'Warn',
								'Number of cycles in the original signal is already lower then the requested minimum number of cycles.'
								)
		#es.JournalWarn('Number of cycles in the original signal is already lower then the requested minimum number of cycles.' )
		return [None, None, messages]
	scale_factor=1.0	
	CyclesInEquivalentSignal=0
	Equivalent_signal_MaxReduction=[]
	
	Equivalent_signal=[]

	#for r in range(10):
	#	es.JournalOut(str(   RainFlow[r]   ))
	TotalDamageEquiv=0
	#Blocks[-1]=Blocks[-1]*Decimal(1.000001)
	for b in range(len(BlocksIndexed)-1):
		DamageOfBlock=0
		sum_of_maxes=0
		sum_of_means=0
		number_of_cyclesE=0
		
		#for r in RainFlow:
			#if r[2] > Blocks[b] and r[2] <= Blocks[b+1]:
		for r in range(len(RainFlow)):
			if r > BlocksIndexed[b] and r <= BlocksIndexed[b+1]:				
				
				DamageOfBlock=DamageOfBlock+RainFlow[r][1]*TestProgRepetitions
				Range=RainFlow[r][0]
				sum_of_maxes=sum_of_maxes+RainFlow[r][5]
				sum_of_means=sum_of_means+RainFlow[r][5]-RainFlow[r][0]/2.				
				number_of_cyclesE=number_of_cyclesE+1
		TotalDamageEquiv=TotalDamageEquiv+DamageOfBlock
		Equivalent_signal.append(  [Range,DamageOfBlock,sum_of_means/number_of_cyclesE,0,0,sum_of_means/number_of_cyclesE]  )
	#es.JournalOut('TotalDamageEquiv: '+str(   TotalDamageEquiv/total_damage   ))
	
	

	
	
	
	for e in Equivalent_signal:
		repetitions=e[1]/(e[0]**slope)
		e[3]=repetitions
		e[4]=(e[1]*100/(total_damage*TestProgRepetitions))
		
		

	
	while CyclesInEquivalentSignal<=MinimumNumberOfCycles:
		#### scaling the range
		if scale_factor!=1.0:
			Equivalent_signal[0][0] = Equivalent_signal[0][0]*scale_factor
			for e in range(len(Equivalent_signal)-2):
				if Equivalent_signal[e+1][0]*scale_factor >= (Equivalent_signal_MaxReduction[e][0]+Equivalent_signal_MaxReduction[e+1][0])/2.:
					Equivalent_signal[e+1][0] = Equivalent_signal[e+1][0]*scale_factor

		if Include_increased_block_damage==1:
			MaxBlockRange=max(l[5] for l in RainFlow)-min(l[7] for l in RainFlow)
			Equivalent_signal[-1][0]= MaxBlockRange	
			Equivalent_signal[-1][2]= max(l[5] for l in RainFlow)
			Equivalent_signal[-1][5]= max(l[5] for l in RainFlow)-MaxBlockRange/2.					

		for e in Equivalent_signal:
			repetitions=e[1]/(e[0]**slope)
			e[3]=repetitions
			e[4]=e[1]*100/(total_damage*TestProgRepetitions)



		#es.JournalOut('Equi sig :'+str(   Equivalent_signal    ))
	
		DamageFromAllBocks=0
		for e in Equivalent_signal:
			DamageFromAllBocks=DamageFromAllBocks+e[1]
		#es.JournalOut('DamageFromAllBocks :'+str(   DamageFromAllBocks    ))
		
		CyclesInEquivalentSignal=0
		for e in Equivalent_signal:
			CyclesInEquivalentSignal=CyclesInEquivalentSignal+e[3]
		#CyclesInEquivalentSignal=CyclesInEquivalentSignal*TestProgRepetitions
		
		if scale_factor==1.0:
			## Deep copy of max reduction Equivalent_signal
			for e in Equivalent_signal:
				Equivalent_signal_MaxReduction.append(list(e))

		
		
		
		scale_factor=scale_factor-0.0001
		
		
	if Include_increased_block_damage==0:			
		MaxBlockRange=max(l[5] for l in RainFlow)-min(l[7] for l in RainFlow)
		Equivalent_signal[-1][0]= MaxBlockRange	
		Equivalent_signal[-1][2]= max(l[5] for l in RainFlow)
		Equivalent_signal[-1][5]= max(l[5] for l in RainFlow)-MaxBlockRange/2.	

	messages = messange_maker(es, messages, 'PrintOut',
							'Number of cycles in equivalent signal :'+str(   int(round(CyclesInEquivalentSignal ))   )
							)
	messages = messange_maker(es, messages, 'PrintOut',
							'Percentage reduction of number of the cycles :'+str(   (CyclesInEquivalentSignal-TotalNumberOfCyclesInOriginalSig)*100./TotalNumberOfCyclesInOriginalSig   )
							)
	#es.JournalOut('scale_factor :'+str(   scale_factor    ))	
	#es.JournalOut('Equivalent_signal_MaxReduction :'+str(   Equivalent_signal_MaxReduction    ))	
	#es.JournalOut('Equivalent_signal :'+str(   Equivalent_signal    ))			
	#################################
	#### Checking if min force doesn't exceed the minimum from the signal
	#################################
	#		    			   0		   1				2			3				4						5
	#Equivalent_signal		[range][damage of block][max of block][repetition][percent damage of block][average_mean_for_block]
	
	## Checking min value
	min_of_the_signal=Equivalent_signal[-1][2]-Equivalent_signal[-1][0]

	messages=messange_maker(es, messages, 'PrintOut',
				'Number of cycles in the original signal is already lower then the requested minimum number of cycles.'
				)

	block_number=0
	for e in Equivalent_signal:
		block_number=block_number+1
		if e[5]-e[0]/2.<min_of_the_signal:
			#e[5]=e[5]-(e[5]-e[0]/2.-min_of_the_signal)
			e[5]=e[0]/2.+min_of_the_signal
			messages=messange_maker(es, messages, 'Warn',
						   'Mean of the equivelnt signal block number:'+str(   block_number   )  +' have been changed due to min value of this block lower then overal min value of the original signal.'
						   )



	## Checking max value		
	max_of_the_signal=Equivalent_signal[-1][2]
	#es.JournalOut('max_of_the_signal :'+str(   max_of_the_signal    ))
	block_number=0
	for e in Equivalent_signal:
		block_number=block_number+1
		if e[5]+e[0]/2.>max_of_the_signal:
			#e[5]=e[5]-(e[5]+e[0]/2.-max_of_the_signal)
			e[5]=max_of_the_signal-e[0]/2.
			messages=messange_maker(es, messages, 'Warn',
						   'Mean of the equivelnt signal block number:'+str(   block_number   )  +' have been changed due to max value of this block higher then overal max value of the original signal.'
						   )


	Equivalent_signal=convert_to_dict_Equivalent_signal(Equivalent_signal)
	Equivalent_signal_MaxReduction=convert_to_dict_Equivalent_signal(Equivalent_signal_MaxReduction)
	return [Equivalent_signal, Equivalent_signal_MaxReduction, messages]