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

#ifndef _THYRA_FROSCHPRECONDITIONER_FACTORY_DEF_HPP
#define _THYRA_FROSCHPRECONDITIONER_FACTORY_DEF_HPP

#include "Thyra_FROSchPreconditionerFactory_decl.hpp"

#ifdef HAVE_SHYLU_DDFROSCH_STRATIMIKOS

namespace Thyra {
    
    template <class SC,class LO,class GO,class NO>
    FROSchPreconditionerFactory<SC,LO,GO,NO>::FROSchPreconditionerFactory() :
    ParameterList_(Teuchos::rcp(new ParameterList()))
    {}
    
    template <class SC,class LO,class GO,class NO>
    bool FROSchPreconditionerFactory<SC,LO,GO,NO>::isCompatible(const LinearOpSourceBase<SC>& fwdOpSrc) const
    {
        const Teuchos::RCP<const LinearOpBase<SC> > fwdOp = fwdOpSrc.getOp();
        
        //?!? AH: Scheinbar nur kompatibel, falls Tpetra ODER BlockedOperator
#ifdef HAVE_SHYLU_DDFROSCH_TPETRA
        if (Xpetra::ThyraUtils<SC,LO,GO,NO>::isTpetra(fwdOp)) return true;
#endif
        
        if (Xpetra::ThyraUtils<SC,LO,GO,NO>::isBlockedOperator(fwdOp)) return true;
        
        return false;
    }
    
    template <class SC,class LO,class GO,class NO>
    RCP<PreconditionerBase<SC> > FROSchPreconditionerFactory<SC,LO,GO,NO>::createPrec() const
    {
        return Teuchos::rcp(new DefaultPreconditioner<SC>);
    }
    
    template <class SC,class LO,class GO,class NO>
    void FROSchPreconditionerFactory<SC,LO,GO,NO>::initializePrec(const Teuchos::RCP<const LinearOpSourceBase<SC> >& fwdOpSrc,
                                                                  PreconditionerBase<SC>* prec,
                                                                  const ESupportSolveUse supportSolveUse) const
    {
        //?!? AH: Brauchen wir die ganzen typedefs???
        
        // we are using typedefs here, since we are using objects from different packages (Xpetra, Thyra,...)
        typedef Xpetra::Map<LO,GO,NO> XpMap;
        typedef Xpetra::Operator<SC,LO,GO,NO> XpOp;
        typedef Xpetra::ThyraUtils<SC,LO,GO,NO> XpThyUtils;
        typedef Xpetra::CrsMatrix<SC,LO,GO,NO> XpCrsMat;
        typedef Xpetra::BlockedCrsMatrix<SC,LO,GO,NO> XpBlockedCrsMat;
        typedef Xpetra::Matrix<SC,LO,GO,NO> XpMat;
        typedef Xpetra::MultiVector<SC,LO,GO,NO> XpMultVec;
        typedef Xpetra::MultiVector<double,LO,GO,NO> XpMultVecDouble;
        typedef Thyra::LinearOpBase<SC> ThyLinOpBase;
#ifdef HAVE_SHYLU_DDFROSCH_TPETRA
        typedef MueLu::TpetraOperator<SC,LO,GO,NO> MueTpOp;
        typedef Tpetra::Operator<SC,LO,GO,NO> TpOp;
        typedef Thyra::TpetraLinearOp<SC,LO,GO,NO> ThyTpLinOp;
#endif
        
        // Check precondition
        //?!? AH: Das lassen wir erstmal so
        TEUCHOS_ASSERT(Teuchos::nonnull(fwdOpSrc));
        TEUCHOS_ASSERT(this->isCompatible(*fwdOpSrc));
        TEUCHOS_ASSERT(prec);
        
        // Create a copy, as we may remove some things from the list
        //?!? AH: Insbesondere muss in der Liste gesagt werden, welcher Vk benutzt werden soll
        Teuchos::ParameterList paramList = *ParameterList_;
        
        // Retrieve wrapped concrete Xpetra matrix from FwdOp
        const Teuchos::RCP<const ThyLinOpBase> fwdOp = fwdOpSrc->getOp();
        TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(fwdOp));
        
        // Check whether it is Epetra/Tpetra
        bool bIsEpetra = XpThyUtils::isEpetra(fwdOp);
        bool bIsTpetra = XpThyUtils::isTpetra(fwdOp);
        bool bIsBlocked = XpThyUtils::isBlockedOperator(fwdOp);
        //?!? AH: Ich verstehe diese Abfragen nicht so ganz
        TEUCHOS_TEST_FOR_EXCEPT((bIsEpetra == true && bIsTpetra == true));
        TEUCHOS_TEST_FOR_EXCEPT((bIsEpetra == bIsTpetra) && bIsBlocked == false);
        TEUCHOS_TEST_FOR_EXCEPT((bIsEpetra != bIsTpetra) && bIsBlocked == true);
        
        Teuchos::RCP<XpMat> A = Teuchos::null;
        //?!? AH: Fallunterscheidung Blocked ODER nicht Blocked
        if(bIsBlocked) {
            Teuchos::RCP<const Thyra::BlockedLinearOpBase<SC> > ThyBlockedOp = Teuchos::rcp_dynamic_cast<const Thyra::BlockedLinearOpBase<SC> >(fwdOp);
            TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(ThyBlockedOp));
            
            TEUCHOS_TEST_FOR_EXCEPT(ThyBlockedOp->blockExists(0,0)==false);
            
            Teuchos::RCP<const LinearOpBase<SC> > b00 = ThyBlockedOp->getBlock(0,0);
            TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(b00));
            
            Teuchos::RCP<const XpCrsMat > xpetraFwdCrsMat00 = XpThyUtils::toXpetra(b00);
            TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(xpetraFwdCrsMat00));
            
            // MueLu needs a non-const object as input
            Teuchos::RCP<XpCrsMat> xpetraFwdCrsMatNonConst00 = Teuchos::rcp_const_cast<XpCrsMat>(xpetraFwdCrsMat00);
            TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(xpetraFwdCrsMatNonConst00));
            
            // wrap the forward operator as an Xpetra::Matrix that MueLu can work with
            Teuchos::RCP<XpMat> A00 = Teuchos::rcp(new Xpetra::CrsMatrixWrap<SC,LO,GO,NO>(xpetraFwdCrsMatNonConst00));
            TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(A00));
            
            Teuchos::RCP<const XpMap> rowmap00 = A00->getRowMap();
            Teuchos::RCP<const Teuchos::Comm<int> > comm = rowmap00->getComm();
            
            //?!? AH: Dies ist die relevante Stelle. Eigentlich wird nur der Communicator extrahiert, um XpBlockedCrsMat zu erstellen
            // create a Xpetra::BlockedCrsMatrix which derives from Xpetra::Matrix that MueLu can work with
            Teuchos::RCP<XpBlockedCrsMat> bMat = Teuchos::rcp(new XpBlockedCrsMat(ThyBlockedOp,comm));
            TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(bMat));
            
            // save blocked matrix
            A = bMat;
        } else {
            Teuchos::RCP<const XpCrsMat > xpetraFwdCrsMat = XpThyUtils::toXpetra(fwdOp);
            TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(xpetraFwdCrsMat));
            
            //?!? AH: Hier wird eine XpCrsMat erstellt
            // MueLu needs a non-const object as input
            Teuchos::RCP<XpCrsMat> xpetraFwdCrsMatNonConst = Teuchos::rcp_const_cast<XpCrsMat>(xpetraFwdCrsMat);
            TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(xpetraFwdCrsMatNonConst));
            
            // wrap the forward operator as an Xpetra::Matrix that MueLu can work with
            A = Teuchos::rcp(new Xpetra::CrsMatrixWrap<SC,LO,GO,NO>(xpetraFwdCrsMatNonConst));
        }
        TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(A));
        
        // AH: Bis hier wurde lediglich Teuchos::RCP<XpMat> A gebaut, das kann man wahrscheinlich erstmal so übernehmen
        
        // Retrieve concrete preconditioner object
        const Teuchos::Ptr<DefaultPreconditioner<SC> > defaultPrec = Teuchos::ptr(dynamic_cast<DefaultPreconditioner<SC> *>(prec));
        TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(defaultPrec));
        
        // extract preconditioner operator
        Teuchos::RCP<ThyLinOpBase> thyra_precOp = Teuchos::null;
        thyra_precOp = Teuchos::rcp_dynamic_cast<Thyra::LinearOpBase<SC> >(defaultPrec->getNonconstUnspecifiedPrecOp(), true);
        
        /////////////////////////////////////
        // AH: Ab hier wird es interessant //
        /////////////////////////////////////
        
        // Variable for multigrid hierarchy: either build a new one or reuse the existing hierarchy
        Teuchos::RCP<MueLu::Hierarchy<SC,LO,GO,NO> > H = Teuchos::null;
        
        // make a decision whether to (re)build the multigrid preconditioner or reuse the old one
        // rebuild preconditioner if startingOver == true
        // reuse preconditioner if startingOver == false
        const bool startingOver = (thyra_precOp.is_null() || !paramList.isParameter("reuse: type") || paramList.get<std::string>("reuse: type") == "none");
        
        if (startingOver == true) {
            // AH: HIER WERDEN DIE KOORDINATEN DER PUNKTE AUS DER PARAMETERLISTE EXTRAHIERT
            // AH: Wir brauchen eine eigene Funktion ExtractCoordinatesFromParameterList
            // extract coordinates from parameter list
            Teuchos::RCP<XpMultVecDouble> coordinates = Teuchos::null;
            coordinates = MueLu::Utilities<SC,LO,GO,NO>::ExtractCoordinatesFromParameterList(paramList);
            
            // TODO check for Xpetra or Thyra vectors?
            Teuchos::RCP<XpMultVec> nullspace = Teuchos::null;
            // AH: HIER WIRD DER NULLSPACE ANGEGEBEN
            // AH: Wir brauchen eine eigene Funktion ExtractCoordinatesFromParameterList
#ifdef HAVE_SHYLU_DDFROSCH_TPETRA
            if (bIsTpetra) {
                typedef Tpetra::MultiVector<SC, LO, GO, NO> tMV;
                Teuchos::RCP<tMV> tpetra_nullspace = Teuchos::null;
                if (paramList.isType<Teuchos::RCP<tMV> >("Nullspace")) {
                    tpetra_nullspace = paramList.get<RCP<tMV> >("Nullspace");
                    paramList.remove("Nullspace");
                    nullspace = MueLu::TpetraMultiVector_To_XpetraMultiVector<SC,LO,GO,NO>(tpetra_nullspace);
                    TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(nullspace));
                }
            }
#endif
            // build a new MueLu hierarchy
            H = MueLu::CreateXpetraPreconditioner(A, paramList,coordinates, nullspace);
            
        } else {
            // reuse old MueLu hierarchy stored in MueLu Tpetra/Epetra operator and put in new matrix
            
            // get old MueLu hierarchy
#if defined(HAVE_SHYLU_DDFROSCH_TPETRA)
            if (bIsTpetra) {
                
                RCP<ThyTpLinOp> tpetr_precOp = rcp_dynamic_cast<ThyTpLinOp>(thyra_precOp);
                RCP<MueTpOp>    muelu_precOp = rcp_dynamic_cast<MueTpOp>(tpetr_precOp->getTpetraOperator(),true);
                
                H = muelu_precOp->GetHierarchy();
            }
#endif
            // TODO add the blocked matrix case here...
            
            TEUCHOS_TEST_FOR_EXCEPTION(!H->GetNumLevels(), MueLu::Exceptions::RuntimeError,
                                       "Thyra::FROSchPreconditionerFactory: Hierarchy has no levels in it");
            TEUCHOS_TEST_FOR_EXCEPTION(!H->GetLevel(0)->IsAvailable("A"), MueLu::Exceptions::RuntimeError,
                                       "Thyra::FROSchPreconditionerFactory: Hierarchy has no fine level operator");
            Teuchos::RCP<MueLu::Level> level0 = H->GetLevel(0);
            Teuchos::RCP<XpOp> O0 = level0->Get<RCP<XpOp> >("A");
            Teuchos::RCP<XpMat> A0 = rcp_dynamic_cast<XpMat>(O0);
            
            if (!A0.is_null()) {
                // If a user provided a "number of equations" argument in a parameter list
                // during the initial setup, we must honor that settings and reuse it for
                // all consequent setups.
                A->SetFixedBlockSize(A0->GetFixedBlockSize());
            }
            
            // set new matrix
            level0->Set("A", A);
            
            H->SetupRe();
        }
        
        // wrap hierarchy H in thyraPrecOp
        RCP<ThyLinOpBase > thyraPrecOp = Teuchos::null;
#if defined(HAVE_SHYLU_DDFROSCH_TPETRA)
        if (bIsTpetra) {
            RCP<MueTpOp> muelu_tpetraOp = Teuchos::rcp(new MueTpOp(H));
            TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(muelu_tpetraOp));
            RCP<TpOp> tpOp = Teuchos::rcp_dynamic_cast<TpOp>(muelu_tpetraOp);
            thyraPrecOp = Thyra::createLinearOp<SC,LO,GO,NO>(tpOp);
        }
#endif
        
        if(bIsBlocked) {
            TEUCHOS_TEST_FOR_EXCEPT(Teuchos::nonnull(thyraPrecOp));
            
            typedef MueLu::XpetraOperator<SC,LO,GO,NO> MueXpOp;
            const Teuchos::RCP<MueXpOp> muelu_xpetraOp = Teuchos::rcp(new MueXpOp(H));
            
            Teuchos::RCP<const VectorSpaceBase<SC> > thyraRangeSpace = Xpetra::ThyraUtils<SC,LO,GO,NO>::toThyra(muelu_xpetraOp->getRangeMap());
            Teuchos::RCP<const VectorSpaceBase<SC> > thyraDomainSpace = Xpetra::ThyraUtils<SC,LO,GO,NO>::toThyra(muelu_xpetraOp->getDomainMap());
            
            Teuchos::RCP<Xpetra::Operator<SC,LO,GO,NO> > xpOp = Teuchos::rcp_dynamic_cast<Xpetra::Operator<SC,LO,GO,NO> >(muelu_xpetraOp);
            thyraPrecOp = Thyra::xpetraLinearOp<SC,LO,GO,NO>(thyraRangeSpace, thyraDomainSpace,xpOp);
        }
        
        TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(thyraPrecOp));
        
        defaultPrec->initializeUnspecified(thyraPrecOp);
        
    }
    
    template <class SC,class LO,class GO,class NO>
    void FROSchPreconditionerFactory<SC,LO,GO,NO>::uninitializePrec(PreconditionerBase<SC>* prec,
                     Teuchos::RCP<const LinearOpSourceBase<SC> >* fwdOp,
                     ESupportSolveUse* supportSolveUse) const
    {
        TEUCHOS_ASSERT(prec);
        
        // Retrieve concrete preconditioner object
        const Teuchos::Ptr<DefaultPreconditioner<SC> > defaultPrec = Teuchos::ptr(dynamic_cast<DefaultPreconditioner<SC> *>(prec));
        TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(defaultPrec));
        
        if (fwdOp) {
            // TODO: Implement properly instead of returning default value
            *fwdOp = Teuchos::null;
        }
        
        if (supportSolveUse) {
            // TODO: Implement properly instead of returning default value
            *supportSolveUse = Thyra::SUPPORT_SOLVE_UNSPECIFIED;
        }
        
        defaultPrec->uninitialize();
    }
    
    
    // Overridden from ParameterListAcceptor
    template <class SC,class LO,class GO,class NO>
    void FROSchPreconditionerFactory<SC,LO,GO,NO>::setParameterList(RCP<ParameterList> const& paramList)
    {
        TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(paramList));
        ParameterList_ = paramList;
    }
    
    template <class SC,class LO,class GO,class NO>
    Teuchos::RCP<Teuchos::ParameterList> FROSchPreconditionerFactory<SC,LO,GO,NO>::getNonconstParameterList()
    {
        return ParameterList_;
    }
    
    template <class SC,class LO,class GO,class NO>
    Teuchos::RCP<Teuchos::ParameterList> FROSchPreconditionerFactory<SC,LO,GO,NO>::unsetParameterList()
    {
        Teuchos::RCP<ParameterList> savedParamList = ParameterList_;
        ParameterList_ = Teuchos::null;
        return savedParamList;
    }
    
    template <class SC,class LO,class GO,class NO>
    Teuchos::RCP<const Teuchos::ParameterList> FROSchPreconditionerFactory<SC,LO,GO,NO>::getParameterList() const
    {
        return ParameterList_;
    }
    
    template <class SC,class LO,class GO,class NO>
    Teuchos::RCP<const Teuchos::ParameterList> FROSchPreconditionerFactory<SC,LO,GO,NO>::getValidParameters() const
    {
        static Teuchos::RCP<const ParameterList> validPL;
        
        if (Teuchos::is_null(validPL))
            validPL = Teuchos::rcp(new ParameterList());
        
        return validPL;
    }
    
    // Public functions overridden from Teuchos::Describable
    template <class SC,class LO,class GO,class NO>
    std::string FROSchPreconditionerFactory<SC,LO,GO,NO>::description() const
    {
        return "Thyra::FROSchPreconditionerFactory";
    }
} // namespace Thyra

#endif // HAVE_MUELU_STRATIMIKOS

#endif // ifdef THYRA_MUELU_PRECONDITIONER_FACTORY_DEF_HPP
