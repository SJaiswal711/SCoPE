! ============================================================================
!  Standalone test: which CAMB background path works?
!
!  Modes:
!    1 = Full CAMB baseline (Max_l=2600)       — slow but reference
!    2 = Minimal Cls (Max_l=100, no lensing)   — ~5-10 ms, should work
!    3 = Pure background (WantCls=.false.)     — expected segfault
! ============================================================================
program test_bg
    use iso_c_binding
    use model
    use results, only: CAMBdata, derived_rdrag, derived_zdrag
    use camb, only: CAMB_ReadParams, CAMB_GetResults
    use IniObjects, only: TIniFile
    implicit none

    type(CAMBparams) :: P_base
    type(TIniFile)   :: ini
    character(len=1024) :: errmsg, arg
    logical :: bad
    integer :: mode

    call get_command_argument(1, arg)
    if (len_trim(arg) == 0) then
        print '(A)', 'Usage: ./test_bg <mode>'
        print '(A)', '  1 = Full CAMB (Max_l=2600)'
        print '(A)', '  2 = Minimal Cls (Max_l=100, no lensing)  <-- fixed'
        print '(A)', '  3 = Pure background (WantCls=F)'
        stop
    end if
    read(arg, *) mode

    call ini%Open('camb_settings.ini', bad, .false.)
    if (bad) stop 'FATAL: cannot open camb_settings.ini'
    if (.not. CAMB_ReadParams(P_base, ini, errmsg)) then
        print '(A,A)', 'FATAL: CAMB_ReadParams failed: ', trim(errmsg)
        stop 1
    end if
    call ini%Close()

    select case (mode)
        case (1); call run_mode(1, P_base)
        case (2); call run_mode(2, P_base)
        case (3); call run_mode(3, P_base)
        case default; print '(A,I0)', 'Unknown mode: ', mode
    end select

contains

    subroutine run_mode(m, P_base)
        integer, intent(in) :: m
        type(CAMBparams), intent(in) :: P_base
        type(CAMBparams) :: Pp
        type(CAMBdata)   :: Cc
        real(8) :: t0, t1, rd, zd, H1, DM1, DH1
        integer :: ierr

        Pp = P_base
        Pp%WantDerivedParameters = .true.
        Pp%WantTransfer          = .false.
        Pp%WantTensors           = .false.
        Pp%WantVectors           = .false.

        select case (m)
            case (1)
                Pp%WantCls     = .true.
                Pp%WantScalars = .true.
                Pp%Max_l       = 2600
                Pp%DoLensing   = .true.
                print '(A)', '=== MODE 1: Full CAMB (Max_l=2600, lensing) ==='
            case (2)
                Pp%WantCls     = .true.
                Pp%WantScalars = .true.
                Pp%Max_l       = 100          ! was 2 — memory corruption below this
                Pp%Max_l_tensor = 0
                Pp%DoLensing   = .false.      ! <-- fixes the heap corruption
                print '(A)', '=== MODE 2: Minimal Cls (Max_l=100, no lensing) ==='
            case (3)
                Pp%WantCls     = .false.
                Pp%WantScalars = .false.
                Pp%DoLensing   = .false.
                print '(A)', '=== MODE 3: Pure background (WantCls=F) ==='
                print '(A)', '  If this segfaults, use mode 2 instead'
        end select

        ierr = 0
        call cpu_time(t0)
        call CAMB_GetResults(Cc, Pp, error=ierr)
        call cpu_time(t1)

        if (ierr /= 0) then
            print '(A,I0)', '  CAMB returned error ierr = ', ierr
            return
        end if

        rd  = Cc%ThermoDerivedParams(derived_rdrag)
        zd  = Cc%ThermoDerivedParams(derived_zdrag)
        H1  = Cc%Hofz(1.0d0) * 299792.458d0
        DM1 = Cc%AngularDiameterDistance(1.0d0) * 2.0d0
        DH1 = 299792.458d0 / H1

        print '(A)', '  ----------------------------------------'
        print '(A,F12.3,A)', '  Time:            ', (t1-t0)*1.0d3, ' ms'
        print '(A,F12.5)',   '  rdrag [Mpc]:     ', rd
        print '(A,F12.5)',   '  zdrag:           ', zd
        print '(A,F12.5)',   '  H(1) [km/s/Mpc]: ', H1
        print '(A,F12.5)',   '  D_M(1) [Mpc]:    ', DM1
        print '(A,F12.5)',   '  D_H(1) [Mpc]:    ', DH1
        print '(A)', '  ----------------------------------------'
    end subroutine run_mode

end program test_bg
