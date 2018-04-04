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

#ifndef _THYRA_FROSCHPRECONDITIONER_FACTORY_DECL_HPP
#define _THYRA_FROSCHPRECONDITIONER_FACTORY_DECL_HPP

#include <MueLu_ConfigDefs.hpp>

#ifdef HAVE_FROSCH_STRATIMIKOS

// Stratimikos needs Thyra, so we don't need special guards for Thyra here
#include "Thyra_DefaultPreconditioner.hpp"
#include "Thyra_BlockedLinearOpBase.hpp"
#include "Thyra_XpetraLinearOp.hpp"
#ifdef HAVE_SHYLU_DDFROSCH_TPETRA
#include "Thyra_TpetraLinearOp.hpp"
#include "Thyra_TpetraThyraWrappers.hpp"
#endif
#ifdef HAVE_SHYLU_DDFROSCH_EPETRA
#include "Thyra_EpetraLinearOp.hpp"
#endif

#include "Teuchos_Ptr.hpp"
#include "Teuchos_TestForException.hpp"
#include "Teuchos_Assert.hpp"
#include "Teuchos_Time.hpp"

#include <Xpetra_CrsMatrixWrap.hpp>
#include <Xpetra_CrsMatrix.hpp>
#include <Xpetra_Matrix.hpp>
#include <Xpetra_ThyraUtils.hpp>

#include <MueLu_Hierarchy.hpp>
#include <MueLu_HierarchyManager.hpp>
#include <MueLu_HierarchyUtils.hpp>
#include <MueLu_Utilities.hpp>
#include <MueLu_ParameterListInterpreter.hpp>
#include <MueLu_MLParameterListInterpreter.hpp>
#include <MueLu_MasterList.hpp>
#include <MueLu_XpetraOperator_decl.hpp> // todo fix me
#include <MueLu_CreateXpetraPreconditioner.hpp>
#ifdef HAVE_SHYLU_DDFROSCH_TPETRA
#include <MueLu_TpetraOperator.hpp>
#endif
#ifdef HAVE_SHYLU_DDFROSCH_EPETRA
#include <MueLu_EpetraOperator.hpp>
#endif

#include "Thyra_PreconditionerFactoryBase.hpp"

#include "Kokkos_DefaultNode.hpp"


namespace Thyra {

    /** @brief Concrete preconditioner factory subclass for Thyra based on FROSch.
        @ingroup FROSchAdapters
        Add support for FROSch preconditioners in Thyra. This class provides an interface both
        for Epetra and Tpetra.

        The general implementation only handles Tpetra. For Epetra there is a specialization
        on SC=double, LO=int, GO=int and NO=EpetraNode.
   */
    template <class SC,
              class LO,
              class GO,
              class NO = KokkosClassic::DefaultNode::DefaultNodeType>
    class FROSchPreconditionerFactory : public PreconditionerFactoryBase<SC> {
    
    public:
        
        typedef Teuchos::RCP<Teuchos::ParameterList> ParameterListPtr;
        typedef Teuchos::RCP<const Teuchos::ParameterList> ConstParameterListPtr;
        
        typedef Teuchos::RCP<PreconditionerBase<SC> > PreconditionerBasePtr;
        
        typedef Teuchos::RCP<const LinearOpSourceBase<SC> > LinearOpSourceBasePtr;
      
        FROSchPreconditionerFactory();

        bool isCompatible(const LinearOpSourceBase<SC>& fwdOp) const;

        PreconditionerBasePtr createPrec() const;

        void initializePrec(const LinearOpSourceBasePtr& fwdOp,
                            PreconditionerBase<SC>* prec,
                            const ESupportSolveUse supportSolveUse) const;

        void uninitializePrec(PreconditionerBase<SC>* prec,
                              LinearOpSourceBasePtr* fwdOp,
                              ESupportSolveUse* supportSolveUse) const;

        void setParameterList(const ParameterListPtr& paramList);
    
        ParameterListPtr unsetParameterList();

        ParameterListPtr getNonconstParameterList();
    
        ConstParameterListPtr getParameterList() const;
    
        ConstParameterListPtr getValidParameters() const;
        
        std::string description() const;

  private:

        ParameterListPtr ParameterList_;

  };

//#ifdef HAVE_SHYLU_DDFROSCH_EPETRA
//    /** @brief Concrete preconditioner factory subclass for Thyra based on FROSch.
//        @ingroup FROSchAdapters
//        Add support for FROSch preconditioners in Thyra. This class provides an interface both
//        for Epetra and Tpetra.
//
//        Specialization for Epetra
//     */
//    template <>
//    class FROSchPreconditionerFactory<double,int,int,Xpetra::EpetraNode> : public PreconditionerFactoryBase<double> {
//
//    public:
//
//        typedef Teuchos::RCP<PreconditionerBase<SC> > PreconditionerBasePtr;
//
//        typedef double SC;
//
//        typedef int LO;
//
//        typedef int GO;
//
//        typedef Xpetra::EpetraNode NO;
//
//
//        FROSchPreconditionerFactory()  : ParameterList_(rcp(new ParameterList())) { }
//
//
//        bool isCompatible(const LinearOpSourceBase<SC>& fwdOpSrc) const
//        {
//            const RCP<const LinearOpBase<SC> > fwdOp = fwdOpSrc.getOp();
//
//#ifdef HAVE_SHYLU_DDFROSCH_TPETRA
//            if (Xpetra::ThyraUtils<SC,LO,GO,NO>::isTpetra(fwdOp)) return true;
//#endif
//
//#ifdef HAVE_SHYLU_DDFROSCH_EPETRA
//            if (Xpetra::ThyraUtils<SC,LO,GO,NO>::isEpetra(fwdOp)) return true;
//#endif
//
//            if (Xpetra::ThyraUtils<SC,LO,GO,NO>::isBlockedOperator(fwdOp)) return true;
//
//            return false;
//        }
//
//        Teuchos::RCP<PreconditionerBase<SC> > createPrec() const
//        {
//            return Teuchos::rcp(new DefaultPreconditioner<SC>);
//        }
//
//        // In die _def Datei packen
//        void initializePrec(const Teuchos::RCP<const LinearOpSourceBase<SC> >& fwdOpSrc,
//                            PreconditionerBase<SC>* prec,
//                            const ESupportSolveUse supportSolveUse) const
//        {
//            using Teuchos::rcp_dynamic_cast;
//
//            typedef Xpetra::Map<LO,GO,NO> XpMap;
//            typedef Xpetra::Operator<SC, LO, GO, NO> XpOp;
//            typedef Xpetra::ThyraUtils<SC,LO,GO,NO>       XpThyUtils;
//            typedef Xpetra::CrsMatrix<SC,LO,GO,NO>        XpCrsMat;
//            typedef Xpetra::BlockedCrsMatrix<SC,LO,GO,NO> XpBlockedCrsMat;
//            typedef Xpetra::Matrix<SC,LO,GO,NO>           XpMat;
//            typedef Xpetra::MultiVector<SC,LO,GO,NO>      XpMultVec;
//            typedef Xpetra::MultiVector<double,LO,GO,NO>      XpMultVecDouble;
//            typedef Thyra::LinearOpBase<SC>                                      ThyLinOpBase;
//
//#ifdef HAVE_SHYLU_DDFROSCH_TPETRA
//#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_INT))) || \
//    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_INT))))
//            typedef MueLu::TpetraOperator<SC,LO,GO,NO> MueTpOp;
//            typedef Tpetra::Operator<SC,LO,GO,NO>      TpOp;
//            typedef Thyra::TpetraLinearOp<SC,LO,GO,NO> ThyTpLinOp;
//#endif
//#endif
//#if defined(HAVE_MUELU_EPETRA)
//            typedef MueLu::EpetraOperator                                         MueEpOp;
//            typedef Thyra::EpetraLinearOp                                         ThyEpLinOp;
//#endif
//
//            // Check precondition
//            TEUCHOS_ASSERT(Teuchos::nonnull(fwdOpSrc));
//            TEUCHOS_ASSERT(this->isCompatible(*fwdOpSrc));
//            TEUCHOS_ASSERT(prec);
//
//            // Create a copy, as we may remove some things from the list
//            ParameterList paramList = *ParameterList_;
//
//            // Retrieve wrapped concrete Xpetra matrix from FwdOp
//            const RCP<const ThyLinOpBase> fwdOp = fwdOpSrc->getOp();
//            TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(fwdOp));
//
//            // Check whether it is Epetra/Tpetra
//            bool bIsEpetra  = XpThyUtils::isEpetra(fwdOp);
//            bool bIsTpetra  = XpThyUtils::isTpetra(fwdOp);
//            bool bIsBlocked = XpThyUtils::isBlockedOperator(fwdOp);
//            TEUCHOS_TEST_FOR_EXCEPT((bIsEpetra == true  && bIsTpetra == true));
//            TEUCHOS_TEST_FOR_EXCEPT((bIsEpetra == bIsTpetra) && bIsBlocked == false);
//            TEUCHOS_TEST_FOR_EXCEPT((bIsEpetra != bIsTpetra) && bIsBlocked == true);
//
//            RCP<XpMat> A = Teuchos::null;
//            if(bIsBlocked) {
//                Teuchos::RCP<const Thyra::BlockedLinearOpBase<SC> > ThyBlockedOp =
//                Teuchos::rcp_dynamic_cast<const Thyra::BlockedLinearOpBase<SC> >(fwdOp);
//                TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(ThyBlockedOp));
//
//                TEUCHOS_TEST_FOR_EXCEPT(ThyBlockedOp->blockExists(0,0)==false);
//
//                Teuchos::RCP<const LinearOpBase<SC> > b00 = ThyBlockedOp->getBlock(0,0);
//                TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(b00));
//
//                RCP<const XpCrsMat > xpetraFwdCrsMat00 = XpThyUtils::toXpetra(b00);
//                TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(xpetraFwdCrsMat00));
//
//                // MueLu needs a non-const object as input
//                RCP<XpCrsMat> xpetraFwdCrsMatNonConst00 = Teuchos::rcp_const_cast<XpCrsMat>(xpetraFwdCrsMat00);
//                TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(xpetraFwdCrsMatNonConst00));
//
//                // wrap the forward operator as an Xpetra::Matrix that MueLu can work with
//                RCP<XpMat> A00 = rcp(new Xpetra::CrsMatrixWrap<SC,LO,GO,NO>(xpetraFwdCrsMatNonConst00));
//                TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(A00));
//
//                RCP<const XpMap> rowmap00 = A00->getRowMap();
//                RCP<const Teuchos::Comm< int > > comm = rowmap00->getComm();
//
//                // create a Xpetra::BlockedCrsMatrix which derives from Xpetra::Matrix that MueLu can work with
//                RCP<XpBlockedCrsMat> bMat = Teuchos::rcp(new XpBlockedCrsMat(ThyBlockedOp, comm));
//                TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(bMat));
//
//                // save blocked matrix
//                A = bMat;
//            } else {
//                RCP<const XpCrsMat > xpetraFwdCrsMat = XpThyUtils::toXpetra(fwdOp);
//                TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(xpetraFwdCrsMat));
//
//                // MueLu needs a non-const object as input
//                RCP<XpCrsMat> xpetraFwdCrsMatNonConst = Teuchos::rcp_const_cast<XpCrsMat>(xpetraFwdCrsMat);
//                TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(xpetraFwdCrsMatNonConst));
//
//                // wrap the forward operator as an Xpetra::Matrix that MueLu can work with
//                A = rcp(new Xpetra::CrsMatrixWrap<SC,LO,GO,NO>(xpetraFwdCrsMatNonConst));
//            }
//            TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(A));
//
//            // Retrieve concrete preconditioner object
//            const Teuchos::Ptr<DefaultPreconditioner<SC> > defaultPrec = Teuchos::ptr(dynamic_cast<DefaultPreconditioner<SC> *>(prec));
//            TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(defaultPrec));
//
//            // extract preconditioner operator
//            RCP<ThyLinOpBase> thyra_precOp = Teuchos::null;
//            thyra_precOp = rcp_dynamic_cast<Thyra::LinearOpBase<SC> >(defaultPrec->getNonconstUnspecifiedPrecOp(), true);
//
//            // Variable for multigrid hierarchy: either build a new one or reuse the existing hierarchy
//            RCP<MueLu::Hierarchy<SC,LO,GO,NO> > H = Teuchos::null;
//
//            // make a decision whether to (re)build the multigrid preconditioner or reuse the old one
//            // rebuild preconditioner if startingOver == true
//            // reuse preconditioner if startingOver == false
//            const bool startingOver = (thyra_precOp.is_null() || !paramList.isParameter("reuse: type") || paramList.get<std::string>("reuse: type") == "none");
//
//            if (startingOver == true) {
//                // extract coordinates from parameter list
//                Teuchos::RCP<XpMultVecDouble> coordinates = Teuchos::null;
//                coordinates = MueLu::Utilities<SC,LO,GO,NO>::ExtractCoordinatesFromParameterList(paramList);
//
//                // TODO check for Xpetra or Thyra vectors?
//                RCP<XpMultVec> nullspace = Teuchos::null;
//#ifdef HAVE_MUELU_TPETRA
//                if (bIsTpetra) {
//#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_INT))) || \
//    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_INT))))
//                    typedef Tpetra::MultiVector<SC, LO, GO, NO> tMV;
//                    RCP<tMV> tpetra_nullspace = Teuchos::null;
//                    if (paramList.isType<Teuchos::RCP<tMV> >("Nullspace")) {
//                        tpetra_nullspace = paramList.get<RCP<tMV> >("Nullspace");
//                        paramList.remove("Nullspace");
//                        nullspace = MueLu::TpetraMultiVector_To_XpetraMultiVector<SC,LO,GO,NO>(tpetra_nullspace);
//                        TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(nullspace));
//                    }
//#else
//                    TEUCHOS_TEST_FOR_EXCEPTION(true, MueLu::Exceptions::RuntimeError,
//                                     "Thyra::MueLuPreconditionerFactory: Tpetra does not support GO=int and or EpetraNode.");
//#endif
//                }
//#endif
//#ifdef HAVE_MUELU_EPETRA
//                if (bIsEpetra) {
//                    RCP<Epetra_MultiVector> epetra_nullspace = Teuchos::null;
//                    if (paramList.isType<RCP<Epetra_MultiVector> >("Nullspace")) {
//                        epetra_nullspace = paramList.get<RCP<Epetra_MultiVector> >("Nullspace");
//                        paramList.remove("Nullspace");
//                        RCP<Xpetra::EpetraMultiVectorT<int,NO> > xpEpNullspace = Teuchos::rcp(new Xpetra::EpetraMultiVectorT<int,NO>(epetra_nullspace));
//                        RCP<Xpetra::MultiVector<double,int,int,NO> > xpEpNullspaceMult = rcp_dynamic_cast<Xpetra::MultiVector<double,int,int,NO> >(xpEpNullspace);
//                        nullspace = rcp_dynamic_cast<XpMultVec>(xpEpNullspaceMult);
//                        TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(nullspace));
//                    }
//                }
//#endif
//                // build a new MueLu hierarchy
//                H = MueLu::CreateXpetraPreconditioner(A, paramList, coordinates, nullspace);
//
//            } else {
//                // reuse old MueLu hierarchy stored in MueLu Tpetra/Epetra operator and put in new matrix
//
//                // get old MueLu hierarchy
//#if defined(HAVE_MUELU_TPETRA)
//                if (bIsTpetra) {
//#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_INT))) || \
//    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_INT))))
//                    RCP<ThyTpLinOp> tpetr_precOp = rcp_dynamic_cast<ThyTpLinOp>(thyra_precOp);
//                    RCP<MueTpOp>    muelu_precOp = rcp_dynamic_cast<MueTpOp>(tpetr_precOp->getTpetraOperator(),true);
//
//                    H = muelu_precOp->GetHierarchy();
//#else
//                    TEUCHOS_TEST_FOR_EXCEPTION(true, MueLu::Exceptions::RuntimeError,
//                                     "Thyra::MueLuPreconditionerFactory: Tpetra does not support GO=int and or EpetraNode.");
//#endif
//                }
//#endif
//#if defined(HAVE_MUELU_EPETRA)// && defined(HAVE_MUELU_SERIAL)
//                if (bIsEpetra) {
//                    RCP<ThyEpLinOp> epetr_precOp = rcp_dynamic_cast<ThyEpLinOp>(thyra_precOp);
//                    RCP<MueEpOp>    muelu_precOp = rcp_dynamic_cast<MueEpOp>(epetr_precOp->epetra_op(),true);
//
//                    H = rcp_dynamic_cast<MueLu::Hierarchy<SC,LO,GO,NO> >(muelu_precOp->GetHierarchy());
//                }
//#endif
//                // TODO add the blocked matrix case here...
//
//                TEUCHOS_TEST_FOR_EXCEPTION(!H->GetNumLevels(), MueLu::Exceptions::RuntimeError,
//                                   "Thyra::MueLuPreconditionerFactory: Hierarchy has no levels in it");
//                TEUCHOS_TEST_FOR_EXCEPTION(!H->GetLevel(0)->IsAvailable("A"), MueLu::Exceptions::RuntimeError,
//                                   "Thyra::MueLuPreconditionerFactory: Hierarchy has no fine level operator");
//                RCP<MueLu::Level> level0 = H->GetLevel(0);
//                RCP<XpOp> O0 = level0->Get<RCP<XpOp> >("A");
//                RCP<XpMat> A0 = rcp_dynamic_cast<XpMat>(O0);
//
//                if (!A0.is_null()) {
//                    // If a user provided a "number of equations" argument in a parameter list
//                    // during the initial setup, we must honor that settings and reuse it for
//                    // all consequent setups.
//                    A->SetFixedBlockSize(A0->GetFixedBlockSize());
//                }
//
//                // set new matrix
//                level0->Set("A", A);
//
//                H->SetupRe();
//            }
//
//            // wrap hierarchy H in thyraPrecOp
//            RCP<ThyLinOpBase > thyraPrecOp = Teuchos::null;
//#if defined(HAVE_MUELU_TPETRA)
//            if (bIsTpetra) {
//#if ((defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_OPENMP) && defined(HAVE_TPETRA_INST_INT_INT))) || \
//    (!defined(EPETRA_HAVE_OMP) && (defined(HAVE_TPETRA_INST_SERIAL) && defined(HAVE_TPETRA_INST_INT_INT))))
//                RCP<MueTpOp> muelu_tpetraOp = rcp(new MueTpOp(H));
//                TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(muelu_tpetraOp));
//                RCP<TpOp> tpOp = Teuchos::rcp_dynamic_cast<TpOp>(muelu_tpetraOp);
//                thyraPrecOp = Thyra::createLinearOp<SC, LO, GO, NO>(tpOp);
//#else
//                TEUCHOS_TEST_FOR_EXCEPTION(true, MueLu::Exceptions::RuntimeError,
//                                           "Thyra::MueLuPreconditionerFactory: Tpetra does not support GO=int and or EpetraNode.");
//#endif
//            }
//#endif
//
//#if defined(HAVE_MUELU_EPETRA)
//            if (bIsEpetra) {
//                RCP<MueLu::Hierarchy<double,int,int,Xpetra::EpetraNode> > epetraH =
//                rcp_dynamic_cast<MueLu::Hierarchy<double,int,int,Xpetra::EpetraNode> >(H);
//                TEUCHOS_TEST_FOR_EXCEPTION(Teuchos::is_null(epetraH), MueLu::Exceptions::RuntimeError,
//                                           "Thyra::MueLuPreconditionerFactory: Failed to cast Hierarchy to Hierarchy<double,int,int,Xpetra::EpetraNode>. Epetra runs only on the Serial node.");
//                RCP<MueEpOp> muelu_epetraOp = rcp(new MueEpOp(epetraH));
//                TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(muelu_epetraOp));
//                // attach fwdOp to muelu_epetraOp to guarantee that it will not go away
//                set_extra_data(fwdOp,"IFPF::fwdOp", Teuchos::inOutArg(muelu_epetraOp), Teuchos::POST_DESTROY,false);
//                RCP<ThyEpLinOp> thyra_epetraOp = Thyra::nonconstEpetraLinearOp(muelu_epetraOp, NOTRANS, EPETRA_OP_APPLY_APPLY_INVERSE, EPETRA_OP_ADJOINT_UNSUPPORTED);
//                TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(thyra_epetraOp));
//                thyraPrecOp = rcp_dynamic_cast<ThyLinOpBase>(thyra_epetraOp);
//            }
//#endif
//
//            if(bIsBlocked) {
//                TEUCHOS_TEST_FOR_EXCEPT(Teuchos::nonnull(thyraPrecOp));
//
//                typedef MueLu::XpetraOperator<SC,LO,GO,NO>    MueXpOp;
//                const RCP<MueXpOp> muelu_xpetraOp = rcp(new MueXpOp(H));
//
//                RCP<const VectorSpaceBase<SC> > thyraRangeSpace  = Xpetra::ThyraUtils<SC,LO,GO,NO>::toThyra(muelu_xpetraOp->getRangeMap());
//                RCP<const VectorSpaceBase<SC> > thyraDomainSpace = Xpetra::ThyraUtils<SC,LO,GO,NO>::toThyra(muelu_xpetraOp->getDomainMap());
//
//                RCP <Xpetra::Operator<SC, LO, GO, NO> > xpOp = Teuchos::rcp_dynamic_cast<Xpetra::Operator<SC,LO,GO,NO> >(muelu_xpetraOp);
//                thyraPrecOp = Thyra::xpetraLinearOp<SC, LO, GO, NO>(thyraRangeSpace, thyraDomainSpace,xpOp);
//            }
//
//            TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(thyraPrecOp));
//
//            defaultPrec->initializeUnspecified(thyraPrecOp);
//        }
//
//        /** \brief . */
//        void uninitializePrec(PreconditionerBase<SC>* prec,
//                              Teuchos::RCP<const LinearOpSourceBase<SC> >* fwdOp,
//                              ESupportSolveUse* supportSolveUse
//                              ) const {
//            TEUCHOS_ASSERT(prec);
//
//            // Retrieve concrete preconditioner object
//            const Teuchos::Ptr<DefaultPreconditioner<SC> > defaultPrec = Teuchos::ptr(dynamic_cast<DefaultPreconditioner<SC> *>(prec));
//            TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(defaultPrec));
//
//            if (fwdOp) {
//                // TODO: Implement properly instead of returning default value
//                *fwdOp = Teuchos::null;
//            }
//
//            if (supportSolveUse) {
//                // TODO: Implement properly instead of returning default value
//                *supportSolveUse = Thyra::SUPPORT_SOLVE_UNSPECIFIED;
//            }
//
//            defaultPrec->uninitialize();
//        }
//
//        //@}
//
//        /** @name Overridden from Teuchos::ParameterListAcceptor */
//        //@{
//
//        /** \brief . */
//        void                                          setParameterList(const ParameterListPtr& paramList) {
//            TEUCHOS_TEST_FOR_EXCEPT(Teuchos::is_null(paramList));
//            ParameterList_ = paramList;
//        }
//        /** \brief . */
//        ParameterListPtr          unsetParameterList() {
//            RCP<ParameterList> savedParamList = ParameterList_;
//            ParameterList_ = Teuchos::null;
//            return savedParamList;
//        }
//        /** \brief . */
//        ParameterListPtr          getNonconstParameterList() { return ParameterList_; }
//        /** \brief . */
//        ConstParameterListPtr    getParameterList() const {   return ParameterList_; }
//        /** \brief . */
//        ConstParameterListPtr    getValidParameters() const {
//            static RCP<const ParameterList> validPL;
//
//            if (Teuchos::is_null(validPL))
//                validPL = rcp(new ParameterList());
//
//            return validPL;
//        }
//        //@}
//
//        /** \name Public functions overridden from Describable. */
//        //@{
//
//        /** \brief . */
//        std::string description() const { return "Thyra::MueLuPreconditionerFactory"; }
//
//        // ToDo: Add an override of describe(...) to give more detail!
//        
//        //@}
//
//    private:
//        ParameterListPtr ParameterList_;
//    }; // end specialization for Epetra
//
//#endif // HAVE_MUELU_EPETRA
    
} // namespace Thyra

#endif // #ifdef HAVE_MUELU_STRATIMIKOS

#endif // THYRA_MUELU_PRECONDITIONER_FACTORY_DECL_HPP
