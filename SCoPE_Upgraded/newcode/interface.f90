module camb_iface
  use iso_c_binding
  use model, only: CAMBparams
  use results, only: CAMBdata
  use camb, only: CAMB_ReadParams, CAMB_GetResults
  use IniObjects, only: TIniFile
  use InitialPower, only: TInitialPowerLaw
  use reionization, only: TTanhReionization
  use DarkEnergyInterface, only: TDarkEnergyEqnOfState 
  use model, only: CT_Temp, CT_E, CT_Cross, CT_B
  implicit none
  
contains

  subroutine camb_from_params_(task, lmax_in, scale_in, cl_tt, cl_te, cl_ee, cl_bb) bind(C, name="camb_from_params_")
    use iso_c_binding, only: c_double, c_int
    implicit none

    real(c_double), intent(in)  :: task(*)
    integer(c_int), intent(in)  :: lmax_in
    real(c_double), intent(in)  :: scale_in
    real(c_double) :: omega_m, omega_b
    real(c_double), intent(out) :: cl_tt(*), cl_te(*), cl_ee(*), cl_bb(*)
    
    ! --- TEMPLATE (Saved) ---
    ! Only save the "Read-Only" template loaded from the file.
    type(CAMBparams), save :: P_init 
    logical, save :: is_first_run = .true.
    
    ! --- WORKERS (NOT Saved) ---
    ! REMOVE 'save' here. These must be fresh every time to prevent memory corruption.
    type(CAMBparams) :: P 
    type(CAMBdata)   :: CAMB_Out
    
    integer :: l, lmax_scalar
    integer, parameter :: IDX_TT = CT_Temp, IDX_EE = CT_E, IDX_TE = CT_Cross , IDX_BB = CT_B
    character(len=1024) :: ErrMsg
    type(TIniFile) :: Ini_loader
    logical :: bad
    
    ! --- Initialization (Runs once) ---
    if (is_first_run) then
        call Ini_loader%Open('camb_settings.ini', bad, .false.)
        if (bad) stop 'FATAL ERROR: Could not open camb_settings.ini'
        if (.not. CAMB_ReadParams(P_init, Ini_loader, ErrMsg)) then
            print *, 'Error parsing CAMB settings: ', trim(ErrMsg)
            stop 1
        end if
        call Ini_loader%Close()
        is_first_run = .false.
    endif

    ! Reset P from template (Clean copy)
    P = P_init

    ! =========================================================================
    ! PARAMETER MAPPING
    ! =========================================================================
    
    ! 1. Basic Cosmology
    omega_m = task(1)
    omega_b = task(2)
    P%ombh2 = omega_b
    P%omch2 = omega_m - omega_b
    P%H0    = 100.0_c_double * task(3)
    
    ! 2. Reionization
    select type(RM => P%Reion)
    class is (TTanhReionization)
        RM%optical_depth = task(4)
    end select
    
    ! 3. Initial Power Spectrum
    select type(InitPower => P%InitPower)
    class is (TInitialPowerLaw)
      InitPower%ns = task(5)
      InitPower%As = exp(task(6)) * 1.0e-10_c_double
      InitPower%nrun = task(10) 
    end select

    ! 4. Dark Energy
    select type(DE => P%DarkEnergy)
    class is (TDarkEnergyEqnOfState)
        DE%w_lam = task(7)
        DE%wa    = task(8)
    end select

    ! 5. Neutrinos
    if (task(9) > 0.0_c_double) then
        P%omnuh2 = task(9) / 93.14_c_double 
    else
        P%omnuh2 = 0.0_c_double
    end if
    
    P%Num_Nu_Massless = task(11)

    ! 6. Curvature
    P%omk = task(12)

    ! =========================================================================

    if (lmax_in > P%Max_l) P%Max_l = lmax_in

    ! Run calculation
    ! Since CAMB_Out is local and fresh, this will behave correctly.
    call CAMB_GetResults(CAMB_Out, P)

    lmax_scalar = min(CAMB_Out%CLData%lmax_lensed, lmax_in)
    
    ! Zero out arrays
    do l = 0, lmax_in
      cl_tt(l+1) = 0.0_c_double; cl_te(l+1) = 0.0_c_double
      cl_ee(l+1) = 0.0_c_double; cl_bb(l+1) = 0.0_c_double
    end do
    
    do l = 2, lmax_scalar
      cl_tt(l+1) = CAMB_Out%CLData%CL_lensed(l, IDX_TT) * scale_in
      cl_te(l+1) = CAMB_Out%CLData%CL_lensed(l, IDX_TE) * scale_in
      cl_ee(l+1) = CAMB_Out%CLData%CL_lensed(l, IDX_EE) * scale_in
      cl_bb(l+1) = CAMB_Out%CLData%CL_lensed(l, IDX_BB) * scale_in
    end do
    

  end subroutine camb_from_params_

end module camb_iface
