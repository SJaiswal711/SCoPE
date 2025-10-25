module camb_iface
    use iso_c_binding
    use model, only: CAMBparams
    use results, only: CAMBdata
    use camb, only: CAMB_ReadParams, CAMB_SetDefParams, CAMB_GetResults
    use IniObjects, only: TIniFile
    use InitialPower, only: TInitialPowerLaw
    use reionization, only: TTanhReionization
    ! --- THE MISSING PIECE ---
    use Recombination, only: TRecfast
    implicit none
  
    ! We will store a "master copy" of the parameters after the first run
    type(CAMBparams), save :: P_master_copy
    logical, save :: is_first_run = .true.
  
  contains
  
    subroutine camb_from_params_(task, lmax_in, scale_in, cl_tt, cl_te, cl_ee, cl_bb) bind(C, name="camb_from_params_")
      use iso_c_binding, only: c_double, c_int
      use model, only: CT_Temp, CT_E, CT_Cross, CT_B
      implicit none
  
      real(c_double), intent(in)  :: task(*)
      integer(c_int),   intent(in)  :: lmax_in
      real(c_double), intent(in)  :: scale_in
      real(c_double), intent(out) :: cl_tt(*), cl_te(*), cl_ee(*), cl_bb(*)
  
      ! --- All declarations must be at the top ---
      type(CAMBparams) :: P
      type(CAMBdata)   :: results
      integer :: l, lmax_scalar
      real(c_double) :: ommh2, ombh2, h, tau, ns, logA, As_val
      integer, parameter :: IDX_TT = CT_Temp, IDX_EE = CT_E, IDX_TE = CT_Cross , IDX_BB = CT_B
      character(len=1024) :: ErrMsg
      ! Variables for the one-time file load
      type(TIniFile) :: Ini_loader
      character(len=256) :: settings_file
      logical :: bad
      
      ! --- Executable code begins here ---
      
      if (is_first_run) then
          settings_file = 'camb_settings.ini'
          print *, 'First run: Loading CAMB settings from: ', trim(settings_file)
          
          call Ini_loader%Open(settings_file, bad, .false.)
          if (bad) then
              print *, 'FATAL ERROR: Could not open camb_settings.ini.'
              stop 1
          endif
          
          ErrMsg = ''
          if (.not. CAMB_ReadParams(P_master_copy, Ini_loader, ErrMsg)) then
              print *, 'Error parsing CAMB settings: ', trim(ErrMsg)
              stop 1
          end if
          call Ini_loader%Close()
          
          is_first_run = .false.
          print *, 'CAMB settings loaded and cached successfully.'
      endif
  
      ! On EVERY run, start with the master copy
      P = P_master_copy
      
      ! Load the 6 MCMC parameters from the C code
      ommh2 = task(1)
      ombh2 = task(2)
      h     = task(3)
      tau   = task(4)
      ns    = task(5)
      logA  = task(6)
      As_val = exp(logA + 2.0_c_double * tau) * 1.0e-10_c_double
      
      ! Overwrite the MCMC parameters on the local P object
      P%H0    = 100.0_c_double * h
      P%ombh2 = ombh2
      P%omch2 = ommh2 - ombh2
  
      select type(RM => P%Reion)
      class is (TTanhReionization)
          RM%optical_depth = tau
      end select
      
      select type(InitPower => P%InitPower)
      class is (TInitialPowerLaw)
        InitPower%As = As_val
        InitPower%ns = ns
      end select
      
      if (lmax_in > P%Max_l) P%Max_l = lmax_in
  
      ! Run the calculation with the updated parameters
      call CAMB_GetResults(results, P)
  
      lmax_scalar = min(results%CLData%lmax_lensed, lmax_in)
      
      do l = 0, lmax_in
        cl_tt(l+1) = 0.0_c_double
        cl_te(l+1) = 0.0_c_double
        cl_ee(l+1) = 0.0_c_double
        cl_bb(l+1) = 0.0_c_double
      end do
      
      do l = 0, lmax_scalar
        cl_tt(l+1) = results%CLData%CL_lensed(l, IDX_TT) * scale_in
        cl_te(l+1) = results%CLData%CL_lensed(l, IDX_TE) * scale_in
        cl_ee(l+1) = results%CLData%CL_lensed(l, IDX_EE) * scale_in
        cl_bb(l+1) = results%CLData%CL_lensed(l, IDX_BB) * scale_in
      end do
  
    end subroutine camb_from_params_
  
  end module camb_iface
