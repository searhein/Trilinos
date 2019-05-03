//@HEADER
// ************************************************************************
//
//               ShyLU: Hybrid preconditioner package
//                 Copyright 2012 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Alexander Heinlein (alexander.heinlein@uni-koeln.de)
//
// ************************************************************************
//@HEADER

#ifndef _FROSCH_ALGEBRAICOVERLAPPINGOPERATOR_DEF_HPP
#define _FROSCH_ALGEBRAICOVERLAPPINGOPERATOR_DEF_HPP
#include <FROSch_SubdomainSolver_decl.hpp>
#include <FROSch_AlgebraicOverlappingOperator_decl.hpp>

namespace FROSch {
	
	std::vector<Teuchos::RCP<Teuchos::Time> > initVec (){
		std::vector<Teuchos::RCP<Teuchos::Time> > m;
		m.resize(2);
		for(int i = 0;i<2;i++){
			m.at(i) = Teuchos::null;
		}
		return m;
	}
	template <class SC,class LO,class GO,class NO>
    int AlgebraicOverlappingOperator<SC,LO,GO,NO>::current_level=0;
	
    template <class SC,class LO,class GO,class NO>
    AlgebraicOverlappingOperator<SC,LO,GO,NO>::AlgebraicOverlappingOperator(CrsMatrixPtr k,
                                                                            ParameterListPtr parameterList) :
    OverlappingOperator<SC,LO,GO,NO> (k,parameterList),
	#ifdef FROSch_AlgebraicOverlappingTimers
	BuildOverlappingMatricesTimer(this->level),
	InitOverlappingOperatorTimer(this->level),
	ComputeOverlappingOperatorTimer(this->level)
	#endif
	{
#ifdef FROSch_AlgebraicOverlappingTimers
		/*if(current_level == 0){
			BuildOverlappingMatricesTimer.reserve(2);
			InitOverlappingOperatorTimer.reserve(2);
			ComputeOverlappingOperatorTimer.reserve(2);
		}*/
		for (int i = 0;i<this->level;i++){
			BuildOverlappingMatricesTimer.at(i) = Teuchos::TimeMonitor::getNewCounter("FROSch AlgebraicOverlappingOperator: BuildOverlappingMatrices " + std::to_string(i));
			InitOverlappingOperatorTimer.at(i) =Teuchos::TimeMonitor::getNewCounter("FROSch AlgebraicOverlappingOperator: InitOverlappingOperator " + std::to_string(i));
			ComputeOverlappingOperatorTimer.at(i) =Teuchos::TimeMonitor::getNewCounter("FROSch AlgebraicOverlappingOperator: ComputeOverlappingOperator " + std::to_string(i));
		}
#endif
	   current_level = current_level +1; 
    }
    
    template <class SC,class LO,class GO,class NO>
    int AlgebraicOverlappingOperator<SC,LO,GO,NO>::initialize(int overlap, MapPtr repeatedMap)
    {
        if (repeatedMap.is_null()) {
            repeatedMap = Xpetra::MapFactory<LO,GO,NO>::Build(this->K_->getRangeMap(),1);
        }
		if(this->MpiComm_->getRank() == 0) std::cout<<"In Here "<<current_level<<" \n";
		{
		#ifdef FROSch_AlgebraicOverlappingTimers
		Teuchos::TimeMonitor BuildOverlappingMatricesTimeMonitor(*BuildOverlappingMatricesTimer.at(current_level-1));
		#endif
        this->buildOverlappingMatrices(overlap,repeatedMap);
		}
		{
	   #ifdef FROSch_AlgebraicOverlappingTimers
		Teuchos::TimeMonitor InitOverlappingOperatorTimeMonitor(*InitOverlappingOperatorTimer.at(current_level-1));
		#endif
        this->initializeOverlappingOperator();
        }
        this->IsInitialized_ = true;
        this->IsComputed_ = false;
        return 0; // RETURN VALUE!!!
    }
    
    template <class SC,class LO,class GO,class NO>
    int AlgebraicOverlappingOperator<SC,LO,GO,NO>::compute()
    {
        FROSCH_ASSERT(this->IsInitialized_,"ERROR: AlgebraicOverlappingOperator has to be initialized before calling compute()");
        {
			
		#ifdef FROSch_AlgebraicOverlappingTimers
		Teuchos::TimeMonitor ComputeOverlappingOperatorTimeMonitor(*ComputeOverlappingOperatorTimer.at(current_level-1));
		#endif
		this->computeOverlappingOperator();
        }
        this->IsComputed_ = true;
        return 0; // RETURN VALUE!!!
    }

    template <class SC,class LO,class GO,class NO>
    void AlgebraicOverlappingOperator<SC,LO,GO,NO>::describe(Teuchos::FancyOStream &out,
                                                             const Teuchos::EVerbosityLevel verbLevel) const
    {
        FROSCH_ASSERT(false,"describe() has be implemented properly...");
    }
    
    template <class SC,class LO,class GO,class NO>
    std::string AlgebraicOverlappingOperator<SC,LO,GO,NO>::description() const
    {
        return "Algebraic Overlapping Operator";
    }
    
    template <class SC,class LO,class GO,class NO>
    int AlgebraicOverlappingOperator<SC,LO,GO,NO>::buildOverlappingMatrices(int overlap,
                                                                            MapPtr repeatedMap)
    {
        //if (this->Verbose_) std::cout << "WARNING: Can we just copy the pointers like that without changing the matrix...?\n";

        this->OverlappingMap_ = repeatedMap;
        this->OverlappingMatrix_ = this->K_;
        for (int i=0; i<overlap; i++) {
            ExtendOverlapByOneLayer(this->OverlappingMatrix_,this->OverlappingMap_);
        }

        return 0;
    }
    
}

#endif
