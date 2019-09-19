#-------------------------------------------------------------------------------------------
# PostMatch: combining open-source software with machine-learning to match lists of postal addresses
# Darren Yates, Md Zahidul Islam, Yanchang Zhao, Richi Nayak, Salil Kanhere, Vladimir Estivill-Castro

# postmatch_v5_XGB.R
# Code, Darren Yates, September 2019
#-------------------------------------------------------------------------------------------
library(poster)
library(ngram)
library(stringdist)
library(xgboost)
library(data.table)
library(caret)
loadedModel <- xgb.load("/home/darren/data61/xgb_data61_v18-Sep-19_v6.model")
#---------------------------------------------------------------------------------------------------------------------------------------
addressListParse = function(addressList) {
    library(parallel)                                                 # load parallel library (multithreading)
    library(poster)                                                   # load poster library (libPostal)
    df <- matrix(mclapply(addressList, splitAddress, mc.cleanup = TRUE, mc.cores=detectCores())) # multi-core sub-function 'splitAddress'
}
#---------------------------------------------------------------------------------------------------------------------------------------
# Function - splitAddress(address)
# - splits address into eight (8) normalised fields, recombined into single list

splitAddress <- function(listAddress) {
    dfAddress <- parse_addr(listAddress)                                     # use libPostal/Poster to split address, creates data-frame
    if (is.na(dfAddress$state)) {
        returnOut <- fixMispeltState(listAddress)
        actualState <- returnOut[-1]
        removeState <- returnOut[-2]
        listAddress <- gsub(removeState, actualState, listAddress)
        dfAddress <- parse_addr(listAddress)
    }
    #-----------------------------------------------------------------------------------------------------------
    dfAddress <- ap_normalise(dfAddress)
    if (is.null(dfAddress$suburb) || is.na(dfAddress$suburb)) citySuburb = dfAddress$city else citySuburb = dfAddress$suburb
    col1 <- dfAddress$po_box
    col2 = dfAddress$unit
    col3 = dfAddress$level
    col4 = dfAddress$house_number
    if (is.na(col2) && !is.na(regexpr('/',col4)) && regexpr('/',col4) > 0) { # split the Aussie 'unit/building' style number into components
        col2 = substr(col4, 1, regexpr('/',col4)-1)
        col4 = substr(col4, regexpr('/',col4)+1, nchar(col4))
    }
    if (!is.na(col2) && is.na(col4) && grepl("lot",col2)) {
        col4 <- col2
        col2 <- "NA"
    }
    col5 <- dfAddress$road
    col6 <- citySuburb
    col7 <- dfAddress$state
    col8 <- dfAddress$postal_code
    vec = c(col1, col2, col3, col4, col5, col6, col7, col8)           # create vector out of the eight (8) address field strings
}
#---------------------------------------------------------------------------------------------------------------------------------------
ap_normalise <- function(address) {
    # Po_box, unit, level, house_number assumed to be correct; otherwise, what do we change them to????
    # state
    splitRoad <- address["road"]
    if (!is.na(splitRoad)) {
        shrinkRoad <- unlist(fixRoad(splitRoad))
        normRoad <- unlist(lapply(shrinkRoad, fixRoadError))
        address["road"] = shrinkRoad
        if (!is.null(normRoad)) {
            address["road"] <- normRoad[1]
            temp <- address["city"]
            if (is.null(address["suburb"]) || is.na(address["suburb"])) citySuburb = address["city"] else citySuburb = address["suburb"]
            if (!is.na(citySuburb)) address["suburb"] <- paste(normRoad[2],citySuburb) else address["suburb"] <- normRoad[2]
            address['suburb'] <- fixTown(address['suburb'])
        }
    }
    normState <- fixState(as.character(address["state"]))
    address["state"] = normState
    return (address)
}
#---------------------------------------------------------------------------------------------------------------------------------------
fixState <- function(x) {
    x = tolower(x)
    if (!is.na(x)) {
        if (x == "new south wales")  x = "nsw"
        if (x == "victoria")  x = "vic"
        if (x == "australian capital territory") x = "act"
        if (x == "queensland")  x = "qld"
        if (x == "tasmania")  x = "tas"
        if (x == "south australia")  x = "sa"
        if (x == "western australia")  x = "wa"
        if (x == "northern territory")  x = "nt"
    }
    return (x)
}
#---------------------------------------------------------------------------------------------------------------------------------------
fixRoad <- function(x) {
    splitRoad <- strsplit(as.character(x), " ")
    elements <- c(unlist(splitRoad))
    windex <- seq_along(elements)
    output <- lapply(windex, function(windex) {
        short <- c("ave","av","blvd","cct","cl","cnr","cr","drv","dr","esp","hwy","ln","pde","pl","rd","st","tce")
        if (!is.na(match(elements[windex], short)) && windex > 1) {
            if (elements[windex] == "ave" || elements[windex] == "av") elements[windex] = "avenue"
            if (elements[windex] == "blvd") elements[windex] = "boulevarde"
            if (elements[windex] == "cct") elements[windex] = "circuit"
            if (elements[windex] == "cl") elements[windex] = "close"
            if (elements[windex] == "cnr") elements[windex] = "corner"
            if (elements[windex] == "cr") elements[windex] = "cresent"
            if (elements[windex] == "ct") elements[windex] = "court"
            if (elements[windex] == "drv" || elements[windex] == "dr") elements[windex] = "drive"
            if (elements[windex] == "esp") elements[windex] = "esplanade"
            if (elements[windex] == "hwy") elements[windex] = "highway"
            if (elements[windex] == "ln") elements[windex] = "lane"
            if (elements[windex] == "pde") elements[windex] = "parade"
            if (elements[windex] == "pl") elements[windex] = "place"
            if (elements[windex] == "rd") elements[windex] = "road"
            if (elements[windex] == "st") elements[windex] = "street"
            if (elements[windex] == "tce") elements[windex] = "terrace"
        }
        return (elements[windex])
    })
    return(paste(output,collapse=' '))
}

# if misspelt town ends up in 'street' field, move everything after suffix back to 'suburb' field
#---------------------------------------------------------------------------------------------------------------------------------------
fixRoadError <- function(x) {
    suffix <- c("road","street","court","avenue","place","lane","drive","way","track","close","crescent","trail","highway",
                "terrace","parade","grove","access","circuit","ramp","boulevard","walk","firetrail","rise","break","loop",
                "mews","link","gardens","circle","pass","parkway","freeway"
    )
    output <- lapply(suffix, function(suffix) {
        splitRoad <- strsplit(as.character(x), " ")
        top <- unlist(splitRoad)
        if (!is.na(match(suffix, top)) && match(suffix,top) < length(top) && match(suffix,top) > 1) {
            extra <- top[((match(suffix,top))+1):length(top)]
            strExtra <- paste(extra,collapse=' ')
            correctRoad <- trimws(gsub(strExtra, "",x))
            splitData <-(c(correctRoad, strExtra))
            return(splitData)
        }
    })
}
#---------------------------------------------------------------------------------------------------------------------------------------
fixTown <- function(x) {
    x <- gsub("street","st",x)
}

#---------------------------------------------------------------------------------------------------------------------------------------
fixNumbers <- function(x) {
    return (sapply(strsplit(x,' '),tail,1))
}
#---------------------------------------------------------------------------------------------------------------------------------------
fixMispeltState <- function(x) {
    x <- as.character(x)
    # we use up to 3-grams to cater for 'new south wales' and 'australian capital territory'
    ng <- c(get.ngrams(ngram(x, n=1)),get.ngrams(ngram(x, n=2)),get.ngrams(ngram(x, n=3)))
    lowestScore <- 1000
    edmin <- 1000
    edstate <- ""
    edorig <- ""
    returnValue <- lapply(ng, function(x) {
        listStates <- c("new south wales", "victoria", "queensland", "tasmania", "western australia", "northern territory",
                        "australian capital territory", "south australia", "nsw", "vic", "qld", "tas", "wa", "nt", "act", "sa")

        ed <- lapply(listStates, function(y) {
            edist <- stringdist(x, y, method="dl")/nchar(x)
            if (edist < edmin) {
                edmin <<- edist
                edstate <- y
                edorig <- x
            }
            output <- c(edorig, edstate)
        })
    })
    newout <- unlist(unique(returnValue))
    newout <- newout[newout!=""]
    newout <- c(newout[length(paste(newout))-1],newout[length(paste(newout))])
}
#---------------------------------------------------------------------------------------------------------------------------------------
postmatch <- function(x, y) {
    list1 <- read.table(paste(x,sep=""),header=F,
                           sep=",",stringsAsFactors=F)
    list2 <- read.table(paste(y,sep=""),header=F,
                        sep=",",stringsAsFactors=F)

    list1 <- matrix(unlist(list1),ncol=1)
    list2 <- matrix(unlist(list2),ncol=1)
    output1 <- lapply(list1, function(list1) {
        output2 <- lapply(list2, function(list2) {
            result <- matchornot(list1, list2)
            if (result[1] == "1") return(c(" > ++ MATCHED: ",result[2],result[3]))
            else if (result[1] == "2") return(c(" > ?? PLEASE CHECK: ",result[2],result[3]))
        })
    })
    overall <- matrix(unlist(output1),ncol=3,byrow=T)
    tableHeader <- c("Match status","List1","List2")
    colnames(overall) <- tableHeader
    print(overall)
}
#---------------------------------------------------------------------------------------------------------------------------------------
matchornot <- function(x,y) {
    address1out <- splitAddress(x)
    address2out <- splitAddress(y)
    pair <- data.frame(address1out, address2out)
    convert <- function(x, output) {
        if (is.na(x[1])) x[1] <- "na"
        if (is.na(x[2])) x[2] <- "na"
        output <- stringdist(x[1],x[2],method="jw")
        output <- 1-output
    }
    newrecord <- apply(pair,1,convert)
    newrecord <- setDT(as.list(newrecord))
    modmatrix <- model.matrix(~.+0, data=newrecord)
    dtest <- xgb.DMatrix(data=modmatrix)
    pred <- predict(loadedModel, dtest, type="class")
    if (pred == 0) {
        return (c(0,x,y))
       } else if (pred == 1) {
        return (c(1,x,y))
    } else if (pred == 2) {
        return (c(2,x,y))
    }
}
#---------------------------------------------------------------------------------------------------------------------------------------
address1 <- "12/31 cook st, springwood qld 4127"
address2 <- "31 cooke st sprngwod 4127 qlde"

address1out <- splitAddress(address1)
address2out <- splitAddress(address2)
pair <- data.frame(address1out, address2out)
convert <- function(x, output) {
    if (is.na(x[1])) x[1] <- "na"
    if (is.na(x[2])) x[2] <- "na"
    output <- stringdist(x[1],x[2],method="jw")
    output <- 1-output
}

print(address1out)
print(address2out)
newrecord <- apply(pair,1,convert)
print(newrecord)
newrecord <- setDT(as.list(newrecord))
modmatrix <- model.matrix(~.+0, data=newrecord)
dtest <- xgb.DMatrix(data=modmatrix)

pred <- predict(loadedModel, dtest, type="class")
if (pred == 0) {
    print("=== PREDICTION : -> xx NOT MATCHED")
} else if (pred == 1) {
    print("=== PREDICTION : -> ** MATCHED")
} else {
    print("=== PREDICTION : -> ?? PLEASE CHECK")
}
#---------------------------------------------------------------------------------------------------------------------------------------
#postmatch("/home/darren/data61/LIST1_10_type0.csv","/home/darren/data61/LIST2_10_type0.csv")
