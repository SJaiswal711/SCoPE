module camb_iface
    use iso_c_binding
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    use model, only: CAMBparams, CT_Temp, CT_E, CT_Cross, CT_B, C_Phi
    ! use results, only: CAMBdata
    use results, only: CAMBdata, derived_rdrag, derived_zdrag
    use transfer, only: Transfer_GetMatterPowerData, MatterPowerData_k, MatterPowerData
    use camb, only: CAMB_ReadParams, CAMB_GetResults
    use IniObjects, only: TIniFile
    use InitialPower, only: TInitialPowerLaw, tensor_param_indeptilt
    use DarkEnergyInterface, only: TDarkEnergyEqnOfState
    use DarkEnergyFluid, only: TDarkEnergyFluid
    use reionization, only: TTanhReionization
    ! use Background, only: Hofz, AngularDiameterDistance  ! For camb_background_
    implicit none
  
    type(TIniFile), save :: g_Ini
    type(CAMBparams), save :: P_base
    logical, save :: g_Initialized = .false.
    logical, save :: warned_transfer = .false.
    real(c_double), parameter :: pi = 3.14159265358979323846_c_double
    real(c_double), parameter :: c_km_s = 299792.458_c_double
  
  contains

    ! --- Shared Initialization ---
    subroutine init_camb()
        logical :: bad
        character(len=1024) :: ErrMsg
        if (.not. g_Initialized) then
            call g_Ini%Open('camb_settings.ini', bad, .false.)
            if (bad) stop 'FATAL ERROR: Could not open camb_settings.ini'
            if (.not. CAMB_ReadParams(P_base, g_Ini, ErrMsg)) then
                print *, 'FATAL ERROR parsing CAMB INI or .dat files: ', trim(ErrMsg)
                stop 1
            end if
            g_Initialized = .true.
        end if
    end subroutine init_camb

    ! --- Shared Parameter Mapping ---
    logical function fill_camb_params(P, task)
        type(CAMBparams), intent(inout) :: P
        real(c_double), intent(in) :: task(*)
        real(c_double) :: omega_m, omega_b, mnu, neff
        
        fill_camb_params = .true.
        omega_m = task(1)
        omega_b = task(2)
        
        if (omega_b >= omega_m .or. omega_m <= 0.0_c_double .or. omega_b <= 0.0_c_double) then
            fill_camb_params = .false.
            return
        end if

        P%ombh2 = omega_b
        P%omch2 = omega_m - omega_b
        P%H0    = 100.0_c_double * task(3)

        select type(RM => P%Reion)
        class is (TTanhReionization)
            RM%optical_depth = task(4)
        end select

        select type(InitPower => P%InitPower)
        class is (TInitialPowerLaw)
            InitPower%ns    = task(5)
            InitPower%As    = exp(task(6)) * 1.0e-10_c_double
            InitPower%nrun  = task(11)
            InitPower%tensor_parameterization = tensor_param_indeptilt
            InitPower%r     = task(12)
            InitPower%nt    = task(13)
            InitPower%ntrun = task(14)
        end select

        select type(DE => P%DarkEnergy)
        class is (TDarkEnergyEqnOfState)
            if (.not. DE%use_tabulated_w) then
                if (task(7) + task(8) > 0.0_c_double) then
                    fill_camb_params = .false.; return
                end if
                select type (DE)
                class is (TDarkEnergyFluid)
                    if (task(8) /= 0.0_c_double .and. (1.0_c_double + task(7) < -1.0e-6_c_double .or. &
                        1.0_c_double + task(7) + task(8) < -1.0e-6_c_double)) then
                        fill_camb_params = .false.; return
                    end if
                end select
                DE%w_lam = task(7)
                DE%wa    = task(8)
            end if
        end select

        mnu  = task(9)
        neff = task(10)
        if (mnu > 0.0_c_double) then
            P%omnuh2 = mnu / 93.14_c_double
            P%Num_Nu_Massive          = 1
            P%Nu_mass_eigenstates     = 1
            P%Nu_mass_numbers(1)      = 1
            P%Nu_mass_degeneracies(1) = 1.0_c_double
            P%Nu_mass_fractions(1)    = 1.0_c_double
            P%Num_Nu_Massless         = neff - 1.0_c_double
        else
            P%omnuh2              = 0.0_c_double
            P%Num_Nu_Massive      = 0
            P%Nu_mass_eigenstates = 0
            P%Num_Nu_Massless     = neff
        end if

        P%YHe = task(15)
        P%omk = task(16)
    end function fill_camb_params
  
    ! --- CMB Perturbations ---
    subroutine camb_from_params_(task, lmax_in, scale_in, num_k, k_in, z_pk, &
                                 cl_tt, cl_te, cl_ee, cl_bb, cl_pp, pk_out) &
          bind(C, name="camb_from_params_")
      use iso_c_binding, only: c_double, c_int
      implicit none
      
      real(c_double), intent(in)  :: task(*), scale_in, k_in(*), z_pk
      integer(c_int), intent(in)  :: lmax_in, num_k
      real(c_double), intent(out) :: cl_tt(*), cl_te(*), cl_ee(*), cl_bb(*), cl_pp(*), pk_out(*)
      
      type(CAMBparams) :: P
      type(CAMBdata)   :: CAMB_Out
      type(MatterPowerData) :: MPD
      integer :: l, lmax_scalar, lmin_t, lmax_t, i, ierr, j, jz
      real(c_double) :: my_nan, conv_fac
  
      my_nan = ieee_value(1.0_c_double, ieee_quiet_nan)
      call init_camb()
  
      if (lmax_in < 2) then
          call trigger_nan_rejection()
          return
      end if
  
      P = P_base 
  
      if (num_k > 0 .and. .not. P%WantTransfer) then
          if (.not. warned_transfer) then
              print *, '[Fortran CAMB] WARNING: C requested P(k), but get_transfer = F in INI. Rejecting.'
              warned_transfer = .true.
          end if
          call trigger_nan_rejection()
          return
      end if
  
      if (.not. fill_camb_params(P, task)) then
          call trigger_nan_rejection()
          return
      end if
  
      if (lmax_in > P%Max_l) P%Max_l = lmax_in
  
      ierr = 0
      call CAMB_GetResults(CAMB_Out, P, error=ierr)
      if (ierr /= 0) then
          call trigger_nan_rejection()
          return
      end if
  
      cl_tt(1:lmax_in+1) = 0.0_c_double; cl_te(1:lmax_in+1) = 0.0_c_double
      cl_ee(1:lmax_in+1) = 0.0_c_double; cl_bb(1:lmax_in+1) = 0.0_c_double
      cl_pp(1:lmax_in+1) = 0.0_c_double
      if (num_k > 0) pk_out(1:num_k) = 0.0_c_double
  
      if (.not. allocated(CAMB_Out%CLData%CL_lensed)) then
          call trigger_nan_rejection()
          return
      end if
  
      lmax_scalar = min(CAMB_Out%CLData%lmax_lensed, lmax_in)
      do l = 0, lmax_scalar
        cl_tt(l+1) = CAMB_Out%CLData%CL_lensed(l, CT_Temp)  * scale_in
        cl_te(l+1) = CAMB_Out%CLData%CL_lensed(l, CT_Cross) * scale_in
        cl_ee(l+1) = CAMB_Out%CLData%CL_lensed(l, CT_E)     * scale_in
        cl_bb(l+1) = CAMB_Out%CLData%CL_lensed(l, CT_B)     * scale_in
      end do
  
      if (P%WantTensors .and. allocated(CAMB_Out%CLData%CL_tensor)) then
          lmin_t = max(0, lbound(CAMB_Out%CLData%CL_tensor, 1))
          lmax_t = min(ubound(CAMB_Out%CLData%CL_tensor, 1), lmax_in)
          if (lmin_t <= lmax_t) then
              do l = lmin_t, lmax_t
                cl_tt(l+1) = cl_tt(l+1) + CAMB_Out%CLData%CL_tensor(l, CT_Temp)  * scale_in
                cl_te(l+1) = cl_te(l+1) + CAMB_Out%CLData%CL_tensor(l, CT_Cross) * scale_in
                cl_ee(l+1) = cl_ee(l+1) + CAMB_Out%CLData%CL_tensor(l, CT_E)     * scale_in
                cl_bb(l+1) = cl_bb(l+1) + CAMB_Out%CLData%CL_tensor(l, CT_B)     * scale_in
              end do
          end if
      end if
  
      if (allocated(CAMB_Out%CLData%Cl_scalar) .and. ubound(CAMB_Out%CLData%Cl_scalar, 2) >= C_Phi) then
          do l = 2, min(ubound(CAMB_Out%CLData%Cl_scalar, 1), lmax_in)
              conv_fac = (real(l+1, c_double) / real(l, c_double))**2 / (2.0_c_double * pi)
              cl_pp(l+1) = CAMB_Out%CLData%Cl_scalar(l, C_Phi) * conv_fac
          end do
      end if
      
      if (num_k > 0) then
          jz = 0
          do j = 1, CAMB_Out%CP%Transfer%PK_num_redshifts
              if (abs(CAMB_Out%CP%Transfer%PK_redshifts(j) - z_pk) < 1.0e-4_c_double) jz = j
          end do
          if (jz == 0) then
              call trigger_nan_rejection()
              return
          end if
          call Transfer_GetMatterPowerData(CAMB_Out, CAMB_Out%MT, MPD, itf_only=CAMB_Out%PK_redshifts_index(jz))
          if (.not. allocated(MPD%matpower)) then
              call trigger_nan_rejection()
              return
          end if
          do i = 1, num_k
              if (.not. (k_in(i) > 0.0_c_double)) then
                  call trigger_nan_rejection()
                  return
              end if
              pk_out(i) = MatterPowerData_k(MPD, k_in(i), 1)
          end do
      end if
  
    contains
      subroutine trigger_nan_rejection()
          if (lmax_in >= 0) cl_tt(1:lmax_in+1) = my_nan
      end subroutine trigger_nan_rejection
  
    end subroutine camb_from_params_

    ! --- Fast Background Solver ---
    subroutine camb_background_(task, num_z, z_in, rd_out, zdrag_out, H_out, DM_out, DH_out) &
          bind(C, name="camb_background_")
      use iso_c_binding, only: c_double, c_int
      implicit none
      
      real(c_double), intent(in)  :: task(*), z_in(*)
      integer(c_int), intent(in)  :: num_z
      real(c_double), intent(out) :: rd_out, zdrag_out, H_out(*), DM_out(*), DH_out(*)
      
      type(CAMBparams) :: P
      type(CAMBdata)   :: CAMB_Out
      integer :: i, ierr
  
      call init_camb()
      P = P_base 
  
      if (.not. fill_camb_params(P, task)) then
          rd_out = -1.0_c_double
          return
      end if
  
      ! Turn off all perturbations for speed (~10ms)
      P%WantCls = .false.
      P%WantTransfer = .false.
      P%WantScalars = .false.
      P%WantTensors = .false.
      P%WantVectors = .false.
      P%WantDerivedParameters = .true.
      
      ierr = 0
      call CAMB_GetResults(CAMB_Out, P, error=ierr)
      if (ierr /= 0) then
          rd_out = -1.0_c_double
          return
      end if
      
    !   rd_out = CAMB_Out%Derived%rdrag
    !   zdrag_out = CAMB_Out%Derived%zdrag

      rd_out = CAMB_Out%ThermoDerivedParams(derived_rdrag)
      zdrag_out = CAMB_Out%ThermoDerivedParams(derived_zdrag)
      
      do i = 1, num_z
          ! Hofz returns H(z) in Mpc^-1. Convert to km/s/Mpc.
        !   H_out(i) = Hofz(z_in(i)) * c_km_s
          ! DM = (1+z) * DA. AngularDiameterDistance returns DA in Mpc.
        !   DM_out(i) = AngularDiameterDistance(z_in(i)) * (1.0_c_double + z_in(i))
        H_out(i) = CAMB_Out%Hofz(z_in(i)) * c_km_s
        DM_out(i) = CAMB_Out%AngularDiameterDistance(z_in(i)) * &
            (1.0_c_double + z_in(i))
        DH_out(i) = c_km_s / H_out(i)
      end do
    end subroutine camb_background_
  
end module camb_iface