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

#ifndef _FROSCH_COARSEOPERATOR_DEF_HPP
#define _FROSCH_COARSEOPERATOR_DEF_HPP

#include <FROSch_CoarseOperator_decl.hpp>
namespace FROSch {
	
	template <class SC,class LO,class GO,class NO>
    int CoarseOperator <SC,LO,GO,NO>::current_level=0;
    
    template<class SC,class LO,class GO,class NO>
    CoarseOperator<SC,LO,GO,NO>::CoarseOperator(CrsMatrixPtr k,
                                                ParameterListPtr parameterList) :
    SchwarzOperator<SC,LO,GO,NO> (k,parameterList),
    CoarseSolveComm_ (),
    OnCoarseSolveComm_ (false),
    NumProcsCoarseSolve_ (0),
    CoarseSpace_ (new CoarseSpace<SC,LO,GO,NO>()),
    Phi_ (),
    CoarseMatrix_ (),
    GatheringMaps_ (0),
    CoarseSolveMap_ (),
    CoarseSolveRepeatedMap_ (),
    BlockCoarseDimension_(),
    CoarseSolver_ (),
    DistributionList_ (sublist(parameterList,"Distribution")),
    CoarseSolveExporters_ (0),
    SubdomainConnectGraph_(),
    GraphEntriesList_(),
    kRowMap_()	
#ifdef FROSch_CoarseOperatorTimers
   ,ComputeTimer(this->level), 
	BuildGraphEntriesTimer(this->level) ,
	BuildGraphTimer(this->level),
	BuildElementNodeListTimer(this->level),
	BuildCoarseMatrixTimer(this->level),
	ApplyTimer(this->level),
	ApplyPhiTTimer(this->level),
	ApplyExportTimer(this->level),
    ApplyCoarseSolveTimer(this->level),
    ApplyPhiTimer(this->level),
    ApplyImportTimer(this->level),
	SetUpTimer(this->level),
	BuildCoarseSolveMapTimer(this->level),
	BuildCoarseRepMapTimer(this->level),
	ExportKOTimer(this->level),
	BuildDirectSolvesTimer(this->level)
#endif    
{
	#ifdef FROSch_CoarseOperatorTimers
	for(int i = 0;i<this->level;i++){
		BuildGraphEntriesTimer.at(i) = Teuchos::TimeMonitor::getNewCounter("FROSch CoarsOperator BuildGraphEntries "+std::to_string(i));
		BuildGraphTimer.at(i) = Teuchos::TimeMonitor::getNewCounter("FROSch CoarsOperator BuildGraph "+std::to_string(i));
		BuildElementNodeListTimer.at(i) = Teuchos::TimeMonitor::getNewCounter("FROSch CoarsOperator BuildElementNodeList "+std::to_string(i));
		ComputeTimer.at(i) = Teuchos::TimeMonitor::getNewCounter("FROSch CoarsOperator Compute "+std::to_string(i));
		ApplyTimer.at(i) = Teuchos::TimeMonitor::getNewCounter("FROSch CoarsOperator Apply "+std::to_string(i));
		ApplyPhiTTimer.at(i) = Teuchos::TimeMonitor::getNewCounter("FROSch CoarsOperator ApplyPhiT "+std::to_string(i));
		ApplyExportTimer.at(i) = Teuchos::TimeMonitor::getNewCounter("FROSch CoarsOperator Export "+std::to_string(i));
		ApplyCoarseSolveTimer.at(i) = Teuchos::TimeMonitor::getNewCounter("FROSch CoarsOperator ApplyCoarseSolve "+std::to_string(i));
		ApplyPhiTimer.at(i) = Teuchos::TimeMonitor::getNewCounter("FROSch CoarsOperator ApplyPhi "+std::to_string(i));
		ApplyImportTimer.at(i) = Teuchos::TimeMonitor::getNewCounter("FROSch CoarsOperator Import "+std::to_string(i));
		SetUpTimer.at(i) = Teuchos::TimeMonitor::getNewCounter("FROSch CoarsOperator SetUp "+std::to_string(i));
		BuildCoarseSolveMapTimer.at(i) = Teuchos::TimeMonitor::getNewCounter("FROSch CoarsOperator BuildCoarseSolveMap "+std::to_string(i));
		BuildCoarseMatrixTimer.at(i) = Teuchos::TimeMonitor::getNewCounter("FROSch CoarsOperator BuildCoarseMatrix "+std::to_string(i));
		BuildCoarseRepMapTimer.at(i) = Teuchos::TimeMonitor::getNewCounter("FROSch CoarsOperator BuildCoarseRepMap "+std::to_string(i));
		ExportKOTimer.at(i) = Teuchos::TimeMonitor::getNewCounter("FROSch CoarsOperator ExportKo "+std::to_string(i));
        BuildDirectSolvesTimer.at(i) = Teuchos::TimeMonitor::getNewCounter("FROSch CoarsOperator BuildDirectSolves "+std::to_string(i));

	}
	#endif
        current_level = current_level+1;
    }
    
    template<class SC,class LO,class GO,class NO>
    CoarseOperator<SC,LO,GO,NO>::~CoarseOperator()
    {
        CoarseSolver_.reset();
    }
    
    template<class SC,class LO, class GO, class NO>
    int CoarseOperator<SC,LO,GO,NO>::buildGraphEntries(Teuchos::RCP<DDInterface<SC,LO,GO,NO> > theDDInterface_){
        
        Teuchos::RCP<Teuchos::FancyOStream> fancy = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));

        #ifdef FROSch_CoarseOperatorTimers
		Teuchos::TimeMonitor BuildGraphEntriesTimeMonitor(*BuildGraphEntriesTimer.at(current_level-1));
		#endif
        int nSubs = this->MpiComm_->getSize();
        int MyPID  = this->MpiComm_->getRank();
        std::map<GO,int> rep;
        Teuchos::Array<GO> entries;
        GOVec2D conn;
        theDDInterface_->identifyConnectivityEntities();
        EntitySetConstPtr Connect =  theDDInterface_->getConnectivityEntities();
        Connect->buildEntityMap(theDDInterface_->getNodesMap());
        InterfaceEntityPtrVec ConnVec = Connect->getEntityVector();
        GO ConnVecSize = ConnVec.size();
        conn.resize(ConnVecSize);
        if (ConnVecSize>0) {
            for(GO i = 0;i<ConnVecSize;i++) {
                conn[i] = ConnVec[i]->getSubdomainsVector();
                for (int j = 0; j<conn[i].size(); j++) rep.insert(std::pair<GO,int>(conn.at(i).at(j),Connect->getEntityMap()->getComm()->getRank()));
            }
            for (auto& x: rep) {
                entries.push_back(x.first);
            }
        }
        Teuchos::RCP<Xpetra::Map<LO,GO,NO> > GraphMap = Xpetra::MapFactory<LO,GO,NO>::Build(Xpetra::UseTpetra,-1,1,0,this->K_->getMap()->getComm());
       
        UN maxNumElements = -1;
        UN numElementsLocal = entries.size();
        reduceAll(*this->MpiComm_,Teuchos::REDUCE_MAX,numElementsLocal,Teuchos::ptr(&maxNumElements));
        Teuchos::RCP<const Xpetra::Map<LO, GO, NO> > ColMap = Xpetra::MapFactory<LO,GO,NO>::createLocalMap(Xpetra::UseTpetra,maxNumElements,this->MpiComm_);
      
        Teuchos::Array<GO> col_vec(entries.size());
        for (UN i = 0; i<entries.size(); i++) {
            col_vec.at(i) = i;
        }
        //Build(Xpetra::UseTpetra,10,col1(),0,this->K_->getMap()->getComm());
        //this->GraphEntriesList_ =  Teuchos::rcp(new Xpetra::TpetraCrsMatrix<GO>(GraphMap,10));
        GraphEntriesList_ = Xpetra::CrsMatrixFactory<GO,LO,GO,NO>::Build(GraphMap,ColMap,numElementsLocal);
        GraphEntriesList_->insertGlobalValues(GraphMap()->getComm()->getRank(),col_vec(),entries());
        GraphEntriesList_->fillComplete();
        return 0;
        
    }
    
    template <class SC,class LO, class GO,class NO>
    int CoarseOperator<SC,LO,GO,NO>::buildConnectGraph(){
        #ifdef FROSch_CoarseOperatorTimers
		Teuchos::TimeMonitor BuildGraphTimeMonitor(*BuildGraphTimer.at(current_level-1));
		#endif
        Teuchos::RCP<Teuchos::FancyOStream> fancy = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));

        
        int nSubs = this->MpiComm_->getSize();
        int MyPID  = this->MpiComm_->getRank();
        
        const size_t numMyElements = GraphEntriesList_->getRowMap()->getNodeNumElements();
        
        Teuchos::ArrayRCP<long unsigned int> numRowEntries(GraphEntriesList_->getRowMap()->getNodeNumElements());
        for (UN i=0; i<GraphEntriesList_->getRowMap()->getNodeNumElements(); i++) {
            numRowEntries[i] = GraphEntriesList_->getNumEntriesInLocalRow(i);
        }
       
        GOVec RowsCoarseSolve;
        
        if (OnCoarseSolveComm_) {
            int start = (nSubs*(CoarseSolveComm_->getRank()))/NumProcsCoarseSolve_;
            int end = (nSubs*(CoarseSolveComm_->getRank()+1))/NumProcsCoarseSolve_;
            RowsCoarseSolve.resize(end-start);
            for (int i = 0; i<end-start; i++) {
                RowsCoarseSolve[i] = start+i;
            }
        }
        Teuchos::RCP<const Xpetra::Map<LO, GO, NO> > ColMap = GraphEntriesList_->getColMap();
        
        Teuchos::RCP<const Xpetra::Map<LO, GO, NO> > GraphMapP = Xpetra::MapFactory<LO,GO,NO>::Build(Xpetra::UseTpetra,-1,RowsCoarseSolve,0,this->MpiComm_);
        Teuchos::RCP<const Xpetra::Map<LO, GO, NO> > GraphMap2 = Xpetra::MapFactory<LO,GO,NO>::Build(Xpetra::UseTpetra,-1,RowsCoarseSolve,0,CoarseSolveComm_);
       
        
        Teuchos::RCP<Xpetra::Import<LO,GO,NO> > scatter = Xpetra::ImportFactory<LO,GO,NO>::Build(GraphEntriesList_->getMap(),GraphMapP);
       
        UN numElementsLocal = GraphEntriesList_->getGlobalMaxNumRowEntries();
        

        Teuchos::RCP<Xpetra::CrsMatrix<GO,LO,GO,NO> > GraphEntries = Xpetra::CrsMatrixFactory<GO,LO,GO,NO>::Build(GraphMapP,ColMap,numElementsLocal);
        GraphEntries->doImport(*GraphEntriesList_,*scatter,Xpetra::INSERT);
        GraphEntries->fillComplete();
        
        Teuchos::ArrayRCP<long unsigned int> numRowEntriesP(GraphEntries->getRowMap()->getNodeNumElements());
        for (UN i=0; i<GraphEntries->getRowMap()->getNodeNumElements(); i++) {
            numRowEntriesP[i] = GraphEntries->getNumEntriesInLocalRow(i);
        }
        
        
        //Teuchos::RCP<const Xpetra::Map<LO, GO, NO> > GraphMap2 = Xpetra::MapFactory<LO,GO,NO>::Build(Xpetra::UseTpetra,-1,RowsCoarseSolve,0,CoarseSolveComm_);
        
        
        const size_t numMyElementS = GraphMapP->getNodeNumElements();
        
        //GraphEntries->describe(*fancy,Teuchos::VERB_EXTREME);
        
        
        
        Teuchos::RCP<const Xpetra::Map<LO, GO, NO> > ColMapG = Xpetra::MapFactory<LO,GO,NO>::createLocalMap(Xpetra::UseTpetra,nSubs,CoarseSolveComm_);
        if (OnCoarseSolveComm_) {
            //SubdomainConnectGraph_= Xpetra::CrsGraphFactory<LO,GO,NO>::Build(GraphMap2,ColMapG,numRowEntries.getConst());
            SubdomainConnectGraph_= Xpetra::CrsGraphFactory<LO,GO,NO>::Build(GraphMap2,ColMapG,numRowEntriesP.getConst());
            for (size_t k = 0; k<numMyElementS; k++) {
                Teuchos::ArrayView<const LO> in;
                Teuchos::ArrayView<const GO> vals_graph;
                GO kg = GraphMapP->getGlobalElement(k);
                GraphEntries->getLocalRowView(k,in,vals_graph);
                Teuchos::Array<GO> vals(vals_graph);
                SubdomainConnectGraph_->insertGlobalIndices(kg,vals());
            }
            SubdomainConnectGraph_->fillComplete();
            
        }
        
        return 0;
    }
    
    template <class SC,class LO,class GO, class NO>
    int CoarseOperator<SC,LO,GO,NO>::buildElementNodeList(MapPtr &tmpCoarseMap){
        Teuchos::RCP<Teuchos::FancyOStream> fancy = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
		#ifdef FROSch_CoarseOperatorTimers
		Teuchos::TimeMonitor BuildElementNodeListTimeMonitor(*BuildElementNodeListTimer.at(current_level-1));
		#endif
        
        int nSubs = this->MpiComm_->getSize();
        GOVec elementList(tmpCoarseMap->getNodeElementList());
        GOVec RowsCoarseSolve;
        
      
        
        if(OnCoarseSolveComm_) {
            int start = (nSubs*(CoarseSolveComm_->getRank()))/NumProcsCoarseSolve_;
            int end = (nSubs*(CoarseSolveComm_->getRank()+1))/NumProcsCoarseSolve_;
            RowsCoarseSolve.resize(end-start);
            for (int i = 0; i<end-start; i++) {
                RowsCoarseSolve[i] = start+i;
            }
        }
       
        Teuchos::RCP<const Xpetra::Map<LO, GO, NO> > GraphMap = Xpetra::MapFactory<LO,GO,NO>::Build(Xpetra::UseTpetra,-1,RowsCoarseSolve,0,this->MpiComm_);
       
        
        
        Teuchos::RCP<Xpetra::Import<LO,GO,NO> > scatter = Xpetra::ImportFactory<LO,GO,NO>::Build(GraphEntriesList_->getMap(),GraphMap);
        
        Teuchos::ArrayView<const GO> elements_ = kRowMap_->getNodeElementList();
        UN maxNumElements = -1;
        UN numElementsLocal = elements_.size();
        reduceAll(*this->MpiComm_,Teuchos::REDUCE_MAX,numElementsLocal,Teuchos::ptr(&maxNumElements));
        
         Teuchos::RCP<const Xpetra::Map<LO, GO, NO> > ColMap = Xpetra::MapFactory<LO,GO,NO>::createLocalMap(Xpetra::UseTpetra,maxNumElements,this->MpiComm_);
         Teuchos::RCP<Xpetra::CrsMatrix<GO,LO,GO,NO> >  Elem = Xpetra::CrsMatrixFactory<GO,LO,GO,NO>::Build(GraphEntriesList_->getMap(),ColMap,numElementsLocal);
        Teuchos::ArrayView<const GO> myGlobals = GraphEntriesList_->getMap()->getNodeElementList();
       
        
        Teuchos::Array<GO> col_vec(elements_.size());
        for(int i = 0; i<elements_.size(); i++) {
            col_vec.at(i) = i;
        }
        for (size_t i = 0; i < GraphEntriesList_->getMap()->getNodeNumElements(); i++) {
            Elem->insertGlobalValues(myGlobals[i],col_vec(),elements_);
            
        }
        Elem->fillComplete();
        size_t MaxRow = Elem->getGlobalMaxNumRowEntries();
        Teuchos::RCP<Xpetra::Import<LO,GO,NO> > scatter2 = Xpetra::ImportFactory<LO,GO,NO>::Build(Elem->getRowMap(),GraphMap);
        Teuchos::RCP<const Xpetra::Map<LO, GO, NO> > ColMapE = Xpetra::MapFactory<LO,GO,NO>::createLocalMap(Xpetra::UseTpetra,MaxRow,CoarseSolveComm_);
        Teuchos::RCP<Xpetra::CrsMatrix<GO,LO,GO,NO> > ElemS = Xpetra::CrsMatrixFactory<GO,LO,GO,NO>::Build(GraphMap,ColMapE,MaxRow);
        ElemS->doImport(*Elem,*scatter2,Xpetra::INSERT);
        ElemS->fillComplete();
       
        Teuchos::RCP<const Xpetra::Map<LO, GO, NO> > GraphMap2 =Xpetra::MapFactory<LO,GO,NO>::Build(Xpetra::UseTpetra,-1,RowsCoarseSolve,0,CoarseSolveComm_);
        
        
        /*this->MpiComm_->barrier();this->MpiComm_->barrier();this->MpiComm_->barrier();
        if(this->MpiComm_->getRank() == 0) std::cout<<"-----------------------------------------------\n";
        if(OnCoarseSolveComm_) GraphMap2->describe(*fancy,Teuchos::VERB_EXTREME);
        this->MpiComm_->barrier();this->MpiComm_->barrier();this->MpiComm_->barrier();
        if(this->MpiComm_->getRank() == 0) std::cout<<"-----------------------------------------------\n";*/
        ElementNodeList_ = Xpetra::CrsMatrixFactory<GO,LO,GO,NO>::Build(GraphMap2,ColMapE,MaxRow);

        if(OnCoarseSolveComm_){
            const size_t numMyElementS = GraphMap2->getNodeNumElements();
            Teuchos::ArrayView<const GO> myGlobalElements = GraphMap2->getNodeElementList();
            Teuchos::ArrayView<const LO> idEl;
            Teuchos::ArrayView<const GO> va;
            for (UN i = 0; i < numMyElementS; i++) {
                UN en = ElemS->getNumEntriesInLocalRow(i);
                Teuchos::Array<GO> cols2(en);
                for(UN h = 0; h<en; h++) {
                    cols2[h] = h;
                }
            
                ElemS->getLocalRowView(i,idEl,va);
                ElementNodeList_->insertGlobalValues(myGlobalElements[i],cols2,va);
            }
            ElementNodeList_->fillComplete();
        }
        return 0;
        
    }
    
    template <class SC,class LO,class GO,class NO>
    int CoarseOperator<SC,LO,GO,NO>::compute()
    {
		
		#ifdef FROSch_CoarseOperatorTimers
		Teuchos::TimeMonitor ComputeTimeMonitor(*ComputeTimer.at(current_level-1));
		#endif
        FROSCH_ASSERT(this->IsInitialized_,"ERROR: CoarseOperator has to be initialized before calling compute()");
        // This is not optimal yet... Some work could be moved to Initialize
        if (this->Verbose_) {
            std::cerr << "WARNING: Some of the operations could be moved from initialize() to Compute().\n";
        }
        if (!this->ParameterList_->get("Recycling","none").compare("basis") && this->IsComputed_) {
            this->setUpCoarseOperator();
            
            this->IsComputed_ = true;
        } else if(!this->ParameterList_->get("Recycling","none").compare("all") && this->IsComputed_) {
            // Maybe use some advanced settings in the future
        } else {
            clearCoarseSpace(); // AH 12/11/2018: If we do not clear the coarse space, we will always append just append the coarse space
            MapPtr subdomainMap = this->computeCoarseSpace(CoarseSpace_); // AH 12/11/2018: This map could be overlapping, repeated, or unique. This depends on the specific coarse operator
            CoarseSpace_->assembleCoarseSpace();
            CoarseSpace_->buildGlobalBasisMatrix(this->K_->getRangeMap(),subdomainMap,this->ParameterList_->get("Threshold Phi",1.e-8));
            Phi_ = CoarseSpace_->getGlobalBasisMatrix();
            this->setUpCoarseOperator();
            
            this->IsComputed_ = true;
        }
        return 0;
    }
    
    template <class SC,class LO,class GO,class NO>
    int CoarseOperator<SC,LO,GO,NO>::clearCoarseSpace()
    {
        return CoarseSpace_->clearCoarseSpace();
    }
    
    template<class SC,class LO,class GO,class NO>
    void CoarseOperator<SC,LO,GO,NO>::apply(const MultiVector &x,
                                            MultiVector &y,
                                            bool usePreconditionerOnly,
                                            Teuchos::ETransp mode,
                                            SC alpha,
                                            SC beta) const
    {
		#ifdef FROSch_CoarseOperatorTimers
		Teuchos::TimeMonitor ApplyTimeMonitor(*ApplyTimer.at(current_level-1));
		#endif
        static int i = 0;
        if (this->IsComputed_) {
            MultiVectorPtr xTmp = Xpetra::MultiVectorFactory<SC,LO,GO,NO>::Build(x.getMap(),x.getNumVectors());
            *xTmp = x;
            
            if (!usePreconditionerOnly && mode == Teuchos::NO_TRANS) {
                this->K_->apply(x,*xTmp,mode,1.0,0.0);
            }
            
            MultiVectorPtr xCoarseSolve = Xpetra::MultiVectorFactory<SC,LO,GO,NO>::Build(GatheringMaps_[GatheringMaps_.size()-1],x.getNumVectors());
            MultiVectorPtr yCoarseSolve = Xpetra::MultiVectorFactory<SC,LO,GO,NO>::Build(GatheringMaps_[GatheringMaps_.size()-1],y.getNumVectors());
            applyPhiT(*xTmp,*xCoarseSolve);
            applyCoarseSolve(*xCoarseSolve,*yCoarseSolve,mode);
            applyPhi(*yCoarseSolve,*xTmp);
            if (!usePreconditionerOnly && mode != Teuchos::NO_TRANS) {
                this->K_->apply(*xTmp,*xTmp,mode,1.0,0.0);
            }
            y.update(alpha,*xTmp,beta);
        } else {
            if (i==1) {
                if (this->Verbose_) std::cout << "WARNING: CoarseOperator has not been computed yet => It will just act as the identity...\n";
                i++;
            }
            y.update(1.0,x,0.0);
        }
    }
    
    template<class SC,class LO,class GO,class NO>
    void CoarseOperator<SC,LO,GO,NO>::applyPhiT(MultiVector& x,
                                                MultiVector& y) const
    {
		#ifdef FROSch_CoarseOperatorTimers
		Teuchos::TimeMonitor ApplyPhiTTimeMonitor(*ApplyPhiTTimer.at(current_level-1));
		#endif
        MultiVectorPtr xCoarse = Xpetra::MultiVectorFactory<SC,LO,GO,NO>::Build(CoarseSpace_->getBasisMap(),x.getNumVectors());
        
        Phi_->apply(x,*xCoarse,Teuchos::TRANS);
        
        MultiVectorPtr xCoarseSolveTmp;
        for (UN j=0; j<GatheringMaps_.size(); j++) {
            xCoarseSolveTmp = Xpetra::MultiVectorFactory<SC,LO,GO,NO>::Build(GatheringMaps_[j],x.getNumVectors());
			{
				#ifdef FROSch_CoarseOperatorTimers
				Teuchos::TimeMonitor  ApplyExportTimeMonitor(*ApplyExportTimer.at(current_level-1));
				#endif
				xCoarseSolveTmp->doExport(*xCoarse,*CoarseSolveExporters_[j],Xpetra::ADD);
				xCoarse = xCoarseSolveTmp;
			}
        }
        y = *xCoarseSolveTmp;
    }
    
    template<class SC,class LO,class GO,class NO>
    void CoarseOperator<SC,LO,GO,NO>::applyCoarseSolve(MultiVector& x,
                                                       MultiVector& y,
                                                       Teuchos::ETransp mode) const
    {
		#ifdef FROSch_CoarseOperatorTimers
		Teuchos::TimeMonitor ApplyCoarseSolveTimeMonitor(*ApplyCoarseSolveTimer.at(current_level-1));
		#endif
        MultiVectorPtr yTmp;
        if (OnCoarseSolveComm_) {
            x.replaceMap(CoarseSolveMap_);
            yTmp = Xpetra::MultiVectorFactory<SC,LO,GO,NO>::Build(CoarseSolveMap_,x.getNumVectors());
            CoarseSolver_->apply(x,*yTmp,mode);
        } else {
            yTmp = Xpetra::MultiVectorFactory<SC,LO,GO,NO>::Build(CoarseSolveMap_,x.getNumVectors());
        }
        yTmp->replaceMap(GatheringMaps_[GatheringMaps_.size()-1]);
        y = *yTmp;
    }
    
    template<class SC,class LO,class GO,class NO>
    void CoarseOperator<SC,LO,GO,NO>::applyPhi(MultiVector& x,
                                               MultiVector& y) const
    {
		#ifdef FROSch_CoarseOperatorTimers
		Teuchos::TimeMonitor ApplyPhiTimeMonitor(*ApplyPhiTimer.at(current_level-1));
		#endif
        MultiVectorPtr yCoarseSolveTmp = Xpetra::MultiVectorFactory<SC,LO,GO,NO>::Build(x.getMap(),x.getNumVectors());
        *yCoarseSolveTmp = x;

        MultiVectorPtr yCoarse;
        for (int j=GatheringMaps_.size()-1; j>0; j--) {
            yCoarse = Xpetra::MultiVectorFactory<SC,LO,GO,NO>::Build(GatheringMaps_[j-1],x.getNumVectors());
			{
				#ifdef FROSch_CoarseOperatorTimers
				Teuchos::TimeMonitor ApplyImportTimeMonitor(*ApplyImportTimer.at(current_level-1));
				#endif
            yCoarse->doImport(*yCoarseSolveTmp,*CoarseSolveExporters_[j],Xpetra::INSERT);
            yCoarseSolveTmp = yCoarse;
			}
        }
     
        yCoarse = Xpetra::MultiVectorFactory<SC,LO,GO,NO>::Build(CoarseSpace_->getBasisMap(),x.getNumVectors());
     
        yCoarse->doImport(*yCoarseSolveTmp,*CoarseSolveExporters_[0],Xpetra::INSERT);
     
        Phi_->apply(*yCoarse,y,Teuchos::NO_TRANS);
     
    }
    
    template<class SC,class LO,class GO,class NO>
    typename CoarseOperator<SC,LO,GO,NO>::CoarseSpacePtr CoarseOperator<SC,LO,GO,NO>::getCoarseSpace() const
    {
        return CoarseSpace_;
    }
    
    template<class SC,class LO,class GO,class NO>
    int CoarseOperator<SC,LO,GO,NO>::setUpCoarseOperator()
    {
        // Build CoarseMatrix_
        #ifdef FROSch_CoarseOperatorTimers
		Teuchos::TimeMonitor SetUpTimeMonitor(*SetUpTimer.at(current_level-1));
		#endif
        CrsMatrixPtr k0 = buildCoarseMatrix();
        kRowMap_ = k0->getMap();
        // Build Map for the coarse solver
        buildCoarseSolveMap(k0);
        GO matrixNumEntry = k0->getGlobalNumEntries();
        GO numRows = k0->getGlobalNumRows();
        GO numCols = k0->getGlobalNumCols();
        //------------------------------------------------------------------------------------------------------------------------
        // Communicate coarse matrix
        if (!DistributionList_->get("Type","linear").compare("linear")) {
            CoarseSolveExporters_[0] = Xpetra::ExportFactory<LO,GO,NO>::Build(CoarseSpace_->getBasisMap(),GatheringMaps_[0]);
            
            CrsMatrixPtr tmpCoarseMatrix = Xpetra::MatrixFactory<SC,LO,GO,NO>::Build(GatheringMaps_[0],k0->getGlobalMaxNumRowEntries());
            
            tmpCoarseMatrix->doExport(*k0,*CoarseSolveExporters_[0],Xpetra::INSERT);

            for (UN j=1; j<GatheringMaps_.size(); j++) {
                tmpCoarseMatrix->fillComplete();
                k0 = tmpCoarseMatrix;
                CoarseSolveExporters_[j] = Xpetra::ExportFactory<LO,GO,NO>::Build(GatheringMaps_[j-1],GatheringMaps_[j]);
                tmpCoarseMatrix = Xpetra::MatrixFactory<SC,LO,GO,NO>::Build(GatheringMaps_[j],k0->getGlobalMaxNumRowEntries());
                
                tmpCoarseMatrix->doExport(*k0,*CoarseSolveExporters_[j],Xpetra::INSERT);
            }
            //------------------------------------------------------------------------------------------------------------------------
            // Matrix to the new communicator
            if (OnCoarseSolveComm_) {
                CoarseMatrix_ = Xpetra::MatrixFactory<SC,LO,GO,NO>::Build(CoarseSolveMap_,k0->getGlobalMaxNumRowEntries());

                ConstGOVecView indices;
                ConstSCVecView values;
                for (UN i=0; i<tmpCoarseMatrix->getNodeNumRows(); i++) {
                    tmpCoarseMatrix->getGlobalRowView(CoarseSolveMap_->getGlobalElement(i),indices,values);
                    if (indices.size()>0) {
                        CoarseMatrix_->insertGlobalValues(CoarseSolveMap_->getGlobalElement(i),indices,values);
                    } else { // Add diagonal unit for zero rows // Todo: Do you we need to sort the coarse matrix "NodeWise"?
                        GOVec indices(1,CoarseSolveMap_->getGlobalElement(i));
                        SCVec values(1,1.0);
                        CoarseMatrix_->insertGlobalValues(CoarseSolveMap_->getGlobalElement(i),indices(),values());
                    }
                    
                }
                CoarseMatrix_->fillComplete(CoarseSolveMap_,CoarseSolveMap_); //Teuchos::RCP<Teuchos::FancyOStream> fancy = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout)); CoarseMatrix_->describe(*fancy,Teuchos::VERB_EXTREME);
                
                CoarseSolver_.reset(new SubdomainSolver<SC,LO,GO,NO>(CoarseMatrix_,sublist(this->ParameterList_,"CoarseSolver")));
                {
					#ifdef FROSch_CoarseOperatorTimers
					Teuchos::TimeMonitor BuildDirectSolvesTimeMonitor(*BuildDirectSolvesTimer.at(current_level-1));
                    #endif
					CoarseSolver_->initialize();
					CoarseSolver_->compute();					
                }
            }
#ifdef HAVE_SHYLU_DDFROSCH_ZOLTAN2
        } else if (!DistributionList_->get("Type","linear").compare("Zoltan2")) {
            //------------------------------------------------------------------------------------------------------------------------
            //coarse matrix already communicated with Zoltan2. Communicate to CoarseSolveComm.
            //------------------------------------------------------------------------------------------------------------------------
            // Matrix to the new communicator
            if (OnCoarseSolveComm_) {
                CoarseMatrix_ = Xpetra::MatrixFactory<SC,LO,GO,NO>::Build(CoarseSolveMap_,k0->getGlobalMaxNumRowEntries());
                ConstLOVecView indices;
                ConstSCVecView values;
                for (UN i=0; i<k0->getNodeNumRows(); i++) {
                    // different sorted maps: CoarseSolveMap_ and k0
                    LO locRow = k0->getRowMap()->getLocalElement(CoarseSolveMap_->getGlobalElement(i));
                    k0->getLocalRowView(locRow,indices,values);
                    if (indices.size()>0) {
                        GOVec indicesGlob(indices.size());
                        for (UN j=0; j<indices.size(); j++) {
                            indicesGlob[j] = k0->getColMap()->getGlobalElement(indices[j]);
                        }
                        CoarseMatrix_->insertGlobalValues(CoarseSolveMap_->getGlobalElement(i),indicesGlob(),values);
                    } else { // Add diagonal unit for zero rows // Todo: Do you we need to sort the coarse matrix "NodeWise"?
                        GOVec indices(1,CoarseSolveMap_->getGlobalElement(i));
                        SCVec values(1,1.0);
                        CoarseMatrix_->insertGlobalValues(CoarseSolveMap_->getGlobalElement(i),indices(),values());
                    }
                    
                }
                
                CoarseMatrix_->fillComplete(CoarseSolveMap_,CoarseSolveMap_); //Teuchos::RCP<Teuchos::FancyOStream> fancy = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout)); CoarseMatrix_->describe(*fancy,Teuchos::VERB_EXTREME);
                
                if (!this->ParameterList_->sublist("CoarseSolver").get("SolverType","Amesos").compare("MueLu")) {
                    CoarseSolver_.reset(new SubdomainSolver<SC,LO,GO,NO>(CoarseMatrix_,sublist(this->ParameterList_,"CoarseSolver"),BlockCoarseDimension_));
                }
                else{
                    CoarseSolver_.reset(new SubdomainSolver<SC,LO,GO,NO>(CoarseMatrix_,sublist(this->ParameterList_,"CoarseSolver")));

                }

                CoarseSolver_->initialize();

                CoarseSolver_->compute();

            }
            //------------------------------------------------------------------------------------------------------------------------
#endif
        } else {
            FROSCH_ASSERT(false,"Distribution Type unknown!");
        }

       
        
        return 0;
    }
    
    template<class SC,class LO,class GO,class NO>
    typename CoarseOperator<SC,LO,GO,NO>::CrsMatrixPtr CoarseOperator<SC,LO,GO,NO>::buildCoarseMatrix()
    {
		#ifdef FROSch_CoarseOperatorTimers
		Teuchos::TimeMonitor BuildCoarseMatrixTimeMonitor(*BuildCoarseMatrixTimer.at(current_level-1));
		#endif
        CrsMatrixPtr k0 = Xpetra::MatrixFactory<SC,LO,GO,NO>::Build(CoarseSpace_->getBasisMap(),CoarseSpace_->getBasisMap()->getNodeNumElements());
        
        if (this->ParameterList_->get("Use Triple MatrixMultiply",false)) {
            Xpetra::TripleMatrixMultiply<SC,LO,GO,NO>::MultiplyRAP(*Phi_,true,*this->K_,false,*Phi_,false,*k0);
        }
        else{
            CrsMatrixPtr tmp = Xpetra::MatrixFactory<SC,LO,GO,NO>::Build(this->K_->getRowMap(),50);
            Xpetra::MatrixMatrix<SC,LO,GO,NO>::Multiply(*this->K_,false,*Phi_,false,*tmp);
            Xpetra::MatrixMatrix<SC,LO,GO,NO>::Multiply(*Phi_,true,*tmp,false,*k0);
        }
        return k0;
    }
    
    template<class SC,class LO,class GO,class NO>
    int CoarseOperator<SC,LO,GO,NO>::buildCoarseSolveMap(CrsMatrixPtr &k0)
    {
		#ifdef FROSch_CoarseOperatorTimers
		Teuchos::TimeMonitor BuildCoarseSolveMapTimeMonitor(*BuildCoarseSolveMapTimer.at(current_level-1));
		#endif
        Teuchos::RCP<Teuchos::FancyOStream> fancy = fancyOStream(Teuchos::rcpFromRef(std::cout)); 

        NumProcsCoarseSolve_ = DistributionList_->get("NumProcs",0);
        double fac = DistributionList_->get("Factor",1.0);
        
        // Redistribute Matrix
        if (NumProcsCoarseSolve_==0) {
            NumProcsCoarseSolve_ = this->MpiComm_->getSize();//Phi->DomainMap().Comm().getSize();
        } else if (NumProcsCoarseSolve_==1) {
            NumProcsCoarseSolve_ = 1;
        } else if (NumProcsCoarseSolve_==-1) {
            NumProcsCoarseSolve_ = int(1+std::max(k0->getGlobalNumRows()/10000,k0->getGlobalNumEntries()/100000));
        } else if (NumProcsCoarseSolve_>1) {
            
        } else if (NumProcsCoarseSolve_<-1) {
            NumProcsCoarseSolve_ = round(pow(1.0*this->MpiComm_->getSize(), 1./(-NumProcsCoarseSolve_)));
        } else {
            FROSCH_ASSERT(false,"This should never happen...");
        }
        
        NumProcsCoarseSolve_ = (LO)  NumProcsCoarseSolve_ * fac;
        if (NumProcsCoarseSolve_<1) {
            NumProcsCoarseSolve_ = 1;
        }
        
        if (NumProcsCoarseSolve_ >= this->MpiComm_->getSize() && DistributionList_->get("Type","linear").compare("Zoltan2")) {
            GatheringMaps_.resize(1);
            CoarseSolveExporters_.resize(1);
            GatheringMaps_[0] = BuildUniqueMap<LO,GO,NO>(Phi_->getColMap()); // DO WE NEED THIS IN ANY CASE???
            return 0;
        }
        //cout << DistributionList_->get("Type","linear") << std::endl;
        if (!DistributionList_->get("Type","linear").compare("linear")) {
            
            int gatheringSteps = DistributionList_->get("GatheringSteps",1);
            GatheringMaps_.resize(gatheringSteps);
            CoarseSolveExporters_.resize(gatheringSteps);
            
            LO numProcsGatheringStep = this->MpiComm_->getSize();
            GO numGlobalIndices = CoarseSpace_->getBasisMap()->getMaxAllGlobalIndex()+1;
            GO numMyRows;
            double gatheringFactor = pow(double(this->MpiComm_->getSize())/double(NumProcsCoarseSolve_),1.0/double(gatheringSteps));
            
            for (int i=0; i<gatheringSteps-1; i++) {
                numMyRows = 0;
                numProcsGatheringStep = LO(numProcsGatheringStep/gatheringFactor);
                //if (this->Verbose_) std::cout << i << " " << numProcsGatheringStep << " " << numGlobalIndices << std::endl;
                if (this->MpiComm_->getRank()%(this->MpiComm_->getSize()/numProcsGatheringStep) == 0 && this->MpiComm_->getRank()/(this->MpiComm_->getSize()/numProcsGatheringStep) < numProcsGatheringStep) {
                    if (this->MpiComm_->getRank()==0) {
                        numMyRows = numGlobalIndices - (numGlobalIndices/numProcsGatheringStep)*(numProcsGatheringStep-1);
                    } else {
                        numMyRows = numGlobalIndices/numProcsGatheringStep;
                    }
                }
                GatheringMaps_[i] = Xpetra::MapFactory<LO,GO,NO>::Build(CoarseSpace_->getBasisMap()->lib(),-1,numMyRows,0,this->MpiComm_);
            }
            
            numMyRows = 0;
            if (this->MpiComm_->getRank()%(this->MpiComm_->getSize()/NumProcsCoarseSolve_) == 0 && this->MpiComm_->getRank()/(this->MpiComm_->getSize()/NumProcsCoarseSolve_) < NumProcsCoarseSolve_) {
                if (this->MpiComm_->getRank()==0) {
                    numMyRows = numGlobalIndices - (numGlobalIndices/NumProcsCoarseSolve_)*(NumProcsCoarseSolve_-1);
                } else {
                    numMyRows = numGlobalIndices/NumProcsCoarseSolve_;
                }
            }
            GatheringMaps_[gatheringSteps-1] = Xpetra::MapFactory<LO,GO,NO>::Build(CoarseSpace_->getBasisMap()->lib(),-1,numMyRows,0,this->MpiComm_);
            //cout << *GatheringMaps_->at(gatheringSteps-1);
            
            //------------------------------------------------------------------------------------------------------------------------
            // Use a separate Communicator for the coarse problem
            MapPtr tmpCoarseMap = GatheringMaps_[GatheringMaps_.size()-1];
            
            if (tmpCoarseMap->getNodeNumElements()>0) {
                OnCoarseSolveComm_=true;
            }

            CoarseSolveComm_ = this->MpiComm_->split(!OnCoarseSolveComm_,this->MpiComm_->getRank());
            CoarseSolveMap_ = Xpetra::MapFactory<LO,GO,NO>::Build(CoarseSpace_->getBasisMap()->lib(),-1,tmpCoarseMap->getNodeElementList(),0,CoarseSolveComm_);
           
#ifdef HAVE_SHYLU_DDFROSCH_ZOLTAN2
        } else if(!DistributionList_->get("Type","linear").compare("Zoltan2")){
            Teuchos::RCP<Teuchos::FancyOStream> fancy = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
            GatheringMaps_.resize(1);
            CoarseSolveExporters_.resize(1);
            GatheringMaps_[0] = Teuchos::rcp_const_cast<Map> (BuildUniqueMap(k0->getRowMap()));
        
            int gatheringSteps = DistributionList_->get("GatheringSteps",1);
            LO numProcsGatheringStep = this->MpiComm_->getSize();
            GO numGlobalIndices = CoarseSpace_->getBasisMap()->getMaxAllGlobalIndex()+1;
            GO numMyRows;
            double gatheringFactor = pow(double(this->MpiComm_->getSize())/double(NumProcsCoarseSolve_),1.0/double(gatheringSteps));

            
            for (int i=0; i<gatheringSteps-1; i++) {
                numMyRows = 0;
                numProcsGatheringStep = LO(numProcsGatheringStep/gatheringFactor);
                //if (this->Verbose_) std::cout << i << " " << numProcsGatheringStep << " " << numGlobalIndices << std::endl;
                if (this->MpiComm_->getRank()%(this->MpiComm_->getSize()/numProcsGatheringStep) == 0 && this->MpiComm_->getRank()/(this->MpiComm_->getSize()/numProcsGatheringStep) < numProcsGatheringStep) {
                    if (this->MpiComm_->getRank()==0) {
                        numMyRows = numGlobalIndices - (numGlobalIndices/numProcsGatheringStep)*(numProcsGatheringStep-1);
                    } else {
                        numMyRows = numGlobalIndices/numProcsGatheringStep;
                    }
                }
                GatheringMaps_[i] = Xpetra::MapFactory<LO,GO,NO>::Build(CoarseSpace_->getBasisMap()->lib(),-1,numMyRows,0,this->MpiComm_);
                
            }
            numMyRows = 0;
            if (this->MpiComm_->getRank()%(this->MpiComm_->getSize()/NumProcsCoarseSolve_) == 0 && this->MpiComm_->getRank()/(this->MpiComm_->getSize()/NumProcsCoarseSolve_) < NumProcsCoarseSolve_) {
                if (this->MpiComm_->getRank()==0) {
                    numMyRows = numGlobalIndices - (numGlobalIndices/NumProcsCoarseSolve_)*(NumProcsCoarseSolve_-1);
                } else {
                    numMyRows = numGlobalIndices/NumProcsCoarseSolve_;
                }
            }
            GatheringMaps_[gatheringSteps-1] = Xpetra::MapFactory<LO,GO,NO>::Build(CoarseSpace_->getBasisMap()->lib(),-1,numMyRows,0,this->MpiComm_);
            // Use a separate Communicator for the coarse problem
            MapPtr tmpCoarseMap = GatheringMaps_[GatheringMaps_.size()-1];
            
            if (tmpCoarseMap->getNodeNumElements()>0) {
                OnCoarseSolveComm_=true;
            }
            GOVec elementList(tmpCoarseMap->getNodeElementList());
            CoarseSolveComm_ = this->MpiComm_->split(!OnCoarseSolveComm_,this->MpiComm_->getRank());
            //CoarseSolveMap_ = Xpetra::MapFactory<LO,GO,NO>::Build(CoarseSpace_->getBasisMap()->lib(),-1,elementList,0,CoarseSolveComm_);
            buildElementNodeList(tmpCoarseMap);
            buildConnectGraph();
            Teuchos::RCP<Xpetra::Map<LO,GO,NO> > UniqueMap;
            GOVec uniEle;
                    
            if(OnCoarseSolveComm_){
				{
					#ifdef FROSch_CoarseOperatorTimers
					Teuchos::TimeMonitor BuildCoarseRepMapTimeMonitor(*BuildCoarseRepMapTimer.at(current_level-1));
					#endif
                CoarseSolveRepeatedMap_ = FROSch::BuildRepMap_Zoltan<SC,LO,GO,NO>(SubdomainConnectGraph_,ElementNodeList_, DistributionList_,CoarseSolveComm_);
                }
                sublist(this->ParameterList_,"CoarseSolver")->set("Repeated Map",CoarseSolveRepeatedMap_);
                Teuchos::ArrayRCP<Teuchos::RCP<Xpetra::Map<LO,GO,NO> > > RepMapVector(1);
                RepMapVector[0] = CoarseSolveRepeatedMap_;
                Teuchos::RCP<Teuchos::ParameterList> pList = sublist(this->ParameterList_,"CoarseSolver");
                
                sublist(this->ParameterList_,"CoarseSolver")->set("Repeated Map Vector",RepMapVector);
                //
                this->ParameterList_->sublist("CoarseSolver").set("Repeated Map Vector",RepMapVector);
                Teuchos::RCP<Xpetra::Map<LO,GO,NO> > MainCoarseMap = BuildMapFromNodeMap(CoarseSolveRepeatedMap_,2,DimensionWise);
                Teuchos::ArrayRCP<Teuchos::RCP<Xpetra::Map<LO,GO,NO> > > MainCoarseMapVector(1);
                MainCoarseMapVector[0] = MainCoarseMap;
                Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Xpetra::Map<LO,GO,NO> > > > dofsMapsVec;
                Teuchos::ArrayRCP<Teuchos::RCP<Xpetra::Map<LO,GO,NO> > > dofsMaps;
                
                Teuchos::ArrayRCP<DofOrdering> dofOrderings(1);
                dofOrderings[0] = DimensionWise;
                Teuchos::ArrayRCP<UN> dofsPerNodeVector(1);
                dofsPerNodeVector[0] = 2
                ;
                //BuildDofMaps(MainCoarseMap,2,DimensionWise,CoarseSolveRepeatedMap_,dofsMaps);
                

                
                BuildDofMapsVec(MainCoarseMapVector,dofsPerNodeVector,dofOrderings,RepMapVector,dofsMapsVec);
                
                sublist(this->ParameterList_,"CoarseSolver")->set("DofOrdering Vector",dofOrderings);
                sublist(this->ParameterList_,"CoarseSolver")->set("DofsPerNode Vector",dofsPerNodeVector);
            
				UniqueMap = FROSch::BuildUniqueMap<LO,GO,NO>(CoarseSolveRepeatedMap_);
                uniEle = UniqueMap->getNodeElementList();
            }
            
            Teuchos::RCP<Xpetra::Map<LO,GO,NO> > tmpMap = Xpetra::MapFactory<LO,GO,NO>::Build(Xpetra::UseTpetra,-1,uniEle,0,this->MpiComm_);;
             //tmpMap->describe(*fancy,Teuchos::VERB_EXTREME);
            CrsMatrixPtr k0Unique = Xpetra::MatrixFactory<SC,LO,GO,NO>::Build(tmpMap,k0->getGlobalMaxNumRowEntries());
                    
            CoarseSolveExporters_[0] = Xpetra::ExportFactory<LO,GO,NO>::Build(CoarseSpace_->getBasisMap(),tmpMap);
			{
				#ifdef FROSch_CoarseOperatorTimers
				Teuchos::TimeMonitor ExportKOTimeMonitor(*ExportKOTimer.at(current_level-1));
				#endif
				k0Unique->doExport(*k0,*CoarseSolveExporters_[0],Xpetra::INSERT);
			}
            k0Unique->fillComplete(GatheringMaps_[0],GatheringMaps_[0]);
                    
            k0 = k0Unique;
                    
            GatheringMaps_[0] = Teuchos::rcp_const_cast<Map>(k0->getRowMap());
            CoarseSolveExporters_[0] = Xpetra::ExportFactory<LO,GO,NO>::Build(CoarseSpace_->getBasisMap(),GatheringMaps_[0]);
                    
           tmpCoarseMap = GatheringMaps_[0];
           elementList = tmpCoarseMap->getNodeElementList();
           CoarseSolveMap_ = Xpetra::MapFactory<LO,GO,NO>::Build(CoarseSpace_->getBasisMap()->lib(),-1,elementList,0,CoarseSolveComm_);
#endif
        } else {
            FROSCH_ASSERT(false,"Distribution type not defined...");
        }
        
        if (this->Verbose_) {
            std::cout << "### ------------------------------ ###" << std::endl;
            std::cout << "### - NumProcs CoarseMatrix : " << NumProcsCoarseSolve_ << std::endl;
            std::cout << "### ------------------------------ ### " << std::endl;
        }
        return 0;
    }
    
}

#endif
