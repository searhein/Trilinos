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

#ifndef _FROSCH_MLSUBSOLVE_DEF_hpp
#define _FROSCH_MLSUBSOLVE_DEF_hpp

#ifdef HAVE_SHYLU_DDFROSCH_STRATIMIKOS

#include <FROSch_MLSubSolve_decl.hpp>

	template<class SC,class LO,class GO,class NO>
     MLSubSolver<SC,LO,GO,NO>::MLSubSolver(CrsMatrixPtr k,
                                                  ParameterListPtr parameterList,
                                                  GOVecPtr blockCoarseSize) :
												K_ (k),
												ParameterList_ (parameterList),
												ThyraSolver_(),
												IsInitialized_ (false),
												IsComputed_ (false)
	{
		Teuchos::RCP<Xpetra::CrsMatrixWrap<SC,LO,GO> > K_wrap = Teuchos::rcp_dynamic_cast<Xpetra::CrsMatrixWrap<SC,LO,GO> >(K_);
		Teuchos::RCP<const Thyra::LinearOpBase<SC> > K_thyra = Xpetra::ThyraUtils<SC,LO,GO,NO>::toThyra(K_wrap->getCrsMatrix());
		Stratimikos::DefaultLinearSolverBuilder linearSolverBuilder;
		ParameterListPtr solverParameterList = sublist(ParameterList_,"Thyra");     

	   //Teuchos::RCP<FROSch::GDSWPreconditioner<SC,LO,GO,NO> > GP(new GDSWPreconditioner<SC,LO,GO,NO>(K_,solverParameterList));
	   // Teuchos::RCP<FROSch::TwoLevelPreconditioner<SC,LO,GO,NO> > TLP(new FROSch::TwoLevelPreconditioner<SC,LO,GO,NO>(K_,solverParameterList));
	    Stratimikos::enableFROSch<LO,GO,NO> (linearSolverBuilder);
		                  
		linearSolverBuilder.setParameterList(parameterList);
		
		PreFacBasePtr pfbFactory = linearSolverBuilder.createPreconditioningStrategy("");
    
		Teuchos::RCP<Thyra::PreconditionerBase<SC> > ThyraPrec = prec(*pfbFactory,K_thyra);
		Teuchos::RCP<const Thyra::LinearOpBase<SC> > LinearPrecOp = ThyraPrec->getUnspecifiedPrecOp();
		
		ThyraSolver_ = Teuchos::rcp_const_cast<Thyra::LinearOpBase<SC> >(LinearPrecOp);
	}


	template<class SC,class LO,class GO,class NO>
     MLSubSolver<SC,LO,GO,NO>::~MLSubSolver()
	{
		ThyraSolver_.reset();
	}
	
	template<class SC,class LO,class GO,class NO>
	int MLSubSolver<SC,LO,GO,NO>::initialize()
	{
		IsInitialized_ = true;
        IsComputed_ = false;
		return 0;
	}
	
	template<class SC,class LO, class GO,class NO>
	int MLSubSolver<SC,LO,GO,NO>::compute()
	{
		 IsComputed_ = true;
		 return 0;
	}
	
	 // Y = alpha * A^mode * X + beta * Y
    template<class SC,class LO,class GO,class NO>
    void MLSubSolver<SC,LO,GO,NO>::apply(const MultiVector &x,
                                             MultiVector &y,
                                             Teuchos::ETransp mode,
                                             SC alpha,
                                             SC beta) const
    {
		 
		 Teuchos::RCP<Xpetra::MultiVector<SC,LO,GO,NO> > yTmp = Xpetra::MultiVectorFactory<SC,LO,GO,NO>::Build(y.getMap(),x.getNumVectors());
         Teuchos::RCP<Thyra::MultiVectorBase<SC> > thyrax = Teuchos::rcp_const_cast<Thyra::MultiVectorBase<SC> > (Xpetra::ThyraUtils<SC,LO,GO,NO>::toThyraMultiVector(yTmp));
         Teuchos::RCP<const Thyra::MultiVectorBase<SC> > thyraB = Xpetra::ThyraUtils<SC,LO,GO,NO>::toThyraMultiVector(rcpFromRef(x));
		 
		 Thyra::apply<double>( *ThyraSolver_,Thyra::NOTRANS, *thyraB, thyrax.ptr());
		 y = Xpetra::ThyraUtils<SC,LO,GO,NO>::toXpetra(thyrax);
		 
	}
	

#endif
#endif