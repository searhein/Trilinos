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

#ifndef _FROSCH_MLSUBSOLVE_DECL_hpp
#define _FROSCH_MLSUBSOLVE_DECL_hpp

#define FROSCH_ASSERT(A,S) if(!(A)) { std::cerr<<"Assertion failed. "<<S<<std::endl; std::cout.flush(); throw std::out_of_range("Assertion.");};


#include <ShyLU_DDFROSch_config.h>

// Thyra includes
/*
#include <Thyra_LinearOpWithSolveBase.hpp>
#include <Thyra_VectorBase.hpp>
#include <Thyra_SolveSupportTypes.hpp>
#include <Thyra_LinearOpWithSolveBase.hpp>
#include <Thyra_LinearOpWithSolveFactoryHelpers.hpp>
#include <Thyra_TpetraLinearOp.hpp>
#include <Thyra_TpetraMultiVector.hpp>
#include <Thyra_TpetraVector.hpp>
#include <Thyra_TpetraThyraWrappers.hpp>
#include <Thyra_VectorBase.hpp>
#include <Thyra_VectorStdOps.hpp>
#ifdef HAVE_SHYLU_DDFROSCH_EPETRA
#include <Thyra_EpetraLinearOp.hpp>
#endif
#include <Thyra_VectorSpaceBase_def.hpp>
#include <Thyra_VectorSpaceBase_decl.hpp>
*/
#ifdef HAVE_SHYLU_DDFROSCH_STRATIMIKOS

#include "Teuchos_Ptr.hpp"
#include "Teuchos_TestForException.hpp"
#include "Teuchos_Assert.hpp"
#include "Teuchos_Time.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ArrayRCP.hpp"
#include "Teuchos_Array.hpp"
#include <Teuchos_XMLParameterListCoreHelpers.hpp>

//Xpetra
#include <Xpetra_CrsMatrixWrap.hpp>
#include <Xpetra_CrsMatrix.hpp>
#include <Xpetra_Matrix.hpp>
#include <Xpetra_ThyraUtils.hpp>
#ifdef HAVE_SHYLU_DDFROSCH_EPETRA
#include <Xpetra_EpetraCrsMatrix.hpp>
#include <Xpetra_EpetraMap.hpp>
#endif

//Epetra
#ifdef HAVE_SHYLU_DDFROSCH_EPETRA
#include <Epetra_Map.h>
#include <Epetra_MpiComm.h>
#include <Epetra_config.h>
#endif

//#include <FROSch_TwoLevelPreconditioner_def.hpp>

// Stratimikos includes
#include <Stratimikos_DefaultLinearSolverBuilder.hpp>
#include <Xpetra_ThyraUtils.hpp>
#include <Stratimikos_FROSchXpetra.hpp>

//Stratimikos

	
    template <class SC = typename Xpetra::Operator<>::scalar_type,
    class LO = typename Xpetra::Operator<SC>::local_ordinal_type,
    class GO = typename Xpetra::Operator<SC, LO>::global_ordinal_type,
    class NO = typename Xpetra::Operator<SC, LO, GO>::node_type>
    class MLSubSolver : public Xpetra::Operator<SC,LO,GO,NO>{
		public:
			typedef Xpetra::Map<LO,GO,NO> Map;
			typedef Teuchos::RCP<Map> MapPtr;
			typedef Teuchos::RCP<const Map> ConstMapPtr;
			typedef Teuchos::ArrayRCP<MapPtr> MapPtrVecPtr;

			typedef Teuchos::ArrayRCP<GO> GOVecPtr;


			typedef Xpetra::Matrix<SC,LO,GO,NO> CrsMatrix;
			typedef Teuchos::RCP<CrsMatrix> CrsMatrixPtr;
	#ifdef HAVE_SHYLU_DDFROSCH_EPETRA
			typedef Epetra_CrsMatrix EpetraCrsMatrix;
			typedef Teuchos::RCP<EpetraCrsMatrix> EpetraCrsMatrixPtr;
	#endif
			typedef Tpetra::CrsMatrix<SC,LO,GO,NO> TpetraCrsMatrix;
			typedef Teuchos::RCP<TpetraCrsMatrix> TpetraCrsMatrixPtr;

			typedef Xpetra::MultiVector<SC,LO,GO,NO> MultiVector;
			typedef Teuchos::RCP<MultiVector> MultiVectorPtr;
			typedef Teuchos::RCP<const MultiVector> ConstMultiVectorPtr;
	#ifdef HAVE_SHYLU_DDFROSCH_EPETRA
			typedef Epetra_MultiVector EpetraMultiVector;
			typedef Teuchos::RCP<EpetraMultiVector> EpetraMultiVectorPtr;
	#endif
			typedef Tpetra::MultiVector<SC,LO,GO,NO> TpetraMultiVector;
			typedef Teuchos::RCP<TpetraMultiVector> TpetraMultiVectorPtr;

			typedef Teuchos::RCP<Teuchos::ParameterList> ParameterListPtr;
		

			typedef typename Teuchos::RCP<Stratimikos::DefaultLinearSolverBuilder> SolverBuilderPtr;
			typedef typename Teuchos::RCP<Thyra::LinearOpWithSolveBase<SC> > ThyraSolveBasePtr;
			typedef typename Teuchos::RCP<Thyra::LinearOpBase<SC> > ThyraOpBasePtr;
			typedef typename Teuchos::RCP<Thyra::PreconditionerFactoryBase<SC> > PreFacBasePtr;
			typedef Xpetra::ThyraUtils<SC,LO,GO,NO>       XpThyUtils;
	
			MLSubSolver(CrsMatrixPtr k,
                        ParameterListPtr parameterList,
                        GOVecPtr blockCoarseSize=Teuchos::null);
						
			virtual ~MLSubSolver();
			
			virtual int initialize();
			
			virtual int compute();
			
			virtual void apply(const MultiVector &x,
                           MultiVector &y,
                           Teuchos::ETransp mode=Teuchos::NO_TRANS,
                           SC alpha=Teuchos::ScalarTraits<SC>::one(),
                           SC beta=Teuchos::ScalarTraits<SC>::zero()) const;
						   
			 //! Get domain map
			virtual ConstMapPtr getDomainMap() const;

			//! Get range map
			virtual ConstMapPtr getRangeMap() const;
			
			bool isInitialized() const;

			//! Get #IsComputed_
			bool isComputed() const;
			
			
	protected:
			
        //! Matrix
        CrsMatrixPtr K_;

        //! Paremter list
        ParameterListPtr ParameterList_;
        ThyraSolveBasePtr ThyraSolver_;
	
	
		 bool IsInitialized_;

        //! Flag to indicated whether this subdomain solver has been setup/computed
         bool IsComputed_;

	};

#endif
#endif