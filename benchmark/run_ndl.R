library(ndl)
library(ndl2)
#serbianUniLat$Outcomes <- serbianUniLat$LemmaCase
#sw <- estimateWeights(cuesOutcomes=serbianUniLat)

event_files <- list.files('events', pattern='.tab.gz', full.names=TRUE)
n <- length(event_files)

full_df = data.frame()
for (r in 1:10) {
	df <- data.frame(event_file=event_files, event_num=numeric(n), `repeats`=r)
	for (i in 1:n) {
		print(file_path <- event_files[i])
	        event_df <- read.delim(gzfile(file_path))
		print(df$event_num[i] <- nrow(event_df))
	
		proc_time <- system.time({
	        	learner <- learnWeightsTabular(gsub("[.]gz$", "", file_path), alpha=0.1, beta=0.1, lambda=1.0, numThreads=1, useExistingFiles=FALSE)
		})
		print(df$`wctime-Rndl2-1thread`[[i]] <- proc_time[3])  # elapsed
	
		proc_time <- system.time({
	        	learner <- learnWeightsTabular(gsub("[.]gz$", "", file_path), alpha=0.1, beta=0.1, lambda=1.0, numThreads=2, useExistingFiles=FALSE)
		})
		print(df$`wctime-Rndl2-4thread`[[i]] <- proc_time[3])  # elapsed
	
		# ignore ndl1 for some: it's order of magnitudes slower
		if (nrow(event_df) < 100000) {
		  proc_time <- system.time({
		    event_df <- read.delim(gzfile(file_path))
		    output <- estimateWeights(cuesOutcomes=event_df) 
		  }, gcFirst = TRUE)
		  print(df$`wctime-Rndl1`[i] <- proc_time[3])  # elapsed
		} else {
		  df$`wctime-Rndl1`[i] <- NaN
		}
		
	}
	full_df <- rbind(full_df, df)
	write.csv(full_df, 'Rndl_result.csv')
}
