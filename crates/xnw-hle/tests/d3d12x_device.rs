//! Plays the title's D3D12.X start-up: device creation through the
//! `d3d12_xs.dll` export, then the device and its children used the way
//! inlined title code uses them (header fields by offset, table slots by
//! offset, the newer layouts the title gets as `ID3D12Device11`), up to
//! three `PresentX` calls. Needs the host GPU through the Agility SDK
//! (`XNW_AGILITY`); skips without it.

use std::ffi::c_void;
use std::time::{Duration, Instant};

use xnw_gpu::d3d12x::layout::*;
use xnw_gpu::d3d12x::{self, HRESULT, Iid, S_OK, This};

const E_NOTIMPL: HRESULT = 0x8000_4001_u32 as i32;

type Create = extern "system" fn(*const c_void, *const u32, *const Iid, *mut This) -> HRESULT;
type Query = extern "system" fn(This, *const Iid, *mut This) -> HRESULT;
type Count = extern "system" fn(This) -> u32;
type Get = extern "system" fn(This, *mut This) -> HRESULT;

/// Slot `offset` of `object`'s table as `F`.
fn at<F: Copy>(object: This, offset: usize) -> F {
    // SAFETY: our objects' tables have every slot the test names, with
    // the signature it names.
    unsafe { d3d12x::slot(object, offset) }
}

/// A field of `object` at `offset`, as inlined title code reads it.
fn field<T: Copy>(object: This, offset: usize) -> T {
    // SAFETY: the offsets read are inside the objects' console layouts.
    unsafe { object.cast::<u8>().add(offset).cast::<T>().read_unaligned() }
}

fn export(name: &str) -> usize {
    xnw_hle::lookup("d3d12_xs.dll", name).unwrap_or_else(|| panic!("d3d12_xs!{name} not exported"))
}

#[test]
fn title_creates_the_device_and_presents_three_frames() {
    let Some(_) = std::env::var_os("XNW_AGILITY").filter(|d| !d.is_empty()) else {
        println!("SKIPPED: set XNW_AGILITY to the Agility SDK folder");
        return;
    };
    xnw_core::init_log(None);
    let null: This = std::ptr::null();

    // Device creation as the title does it: version 2.4, as ID3D12Device11.
    // SAFETY: the export has this signature.
    let create: Create = unsafe { std::mem::transmute(export("D3D12XboxCreateDevice")) };
    let parameters = [d3d12x::CHECKED_VERSION, 0, 0, 0, 0, 0, 0, 0];
    let mut device = null;
    assert_eq!(create(null.cast(), parameters.as_ptr(), &d3d12x::LEVEL_IIDS[8], &mut device), S_OK);
    assert!(!device.is_null());

    // Object header: table, zero bits, a reference; the device's
    // out-of-line byte at +0x18 the title tests before inlining.
    assert_ne!(field::<usize>(device, 0), 0);
    assert_eq!(field::<u32>(device, 8), 0);
    assert!(field::<u32>(device, 0xc) >= 1);
    assert_ne!(field::<u8>(device, 0x18), 0);
    assert_eq!(at::<Count>(device, 0x38)(device), 1, "node count");
    assert_eq!(at::<extern "system" fn() -> HRESULT>(device, 0xf0)(), E_NOTIMPL, "a naming stub fails cleanly");

    // DXGI: the adapter is a Series S, not the host GPU.
    let mut dxgi = null;
    assert_eq!(at::<Query>(device, 0)(device, &DXGI_DEVICE_IIDS[2], &mut dxgi), S_OK);
    let mut adapter = null;
    assert_eq!(at::<Get>(dxgi, 0x38)(dxgi, &mut adapter), S_OK);
    let mut desc = std::mem::MaybeUninit::<AdapterDescription>::uninit();
    assert_eq!(at::<extern "system" fn(This, *mut AdapterDescription) -> HRESULT>(adapter, 0x40)(adapter, desc.as_mut_ptr()), S_OK);
    // SAFETY: written above.
    let desc = unsafe { desc.assume_init() };
    assert_eq!(desc.vendor, 0x1002);
    assert!(String::from_utf16_lossy(&desc.name).starts_with("Xbox Series S"));

    // Feature queries: levels up to 12_2, the console's 16-word options.
    type Check = extern "system" fn(This, u32, *mut c_void, u32) -> HRESULT;
    #[repr(C)]
    struct Levels {
        count: u32,
        requested: *const u32,
        highest: u32,
    }
    let wanted = [0xb000u32, 0xc000, 0xc100, 0xc200, 0xd000];
    let mut levels = Levels { count: 5, requested: wanted.as_ptr(), highest: 0 };
    assert_eq!(at::<Check>(device, 0x68)(device, 2, (&raw mut levels).cast(), 0x18), S_OK);
    assert_eq!(levels.highest, 0xc200);
    let mut options = [0u32; 16];
    assert_eq!(at::<Check>(device, 0x68)(device, 0, options.as_mut_ptr().cast(), 64), S_OK);
    assert_eq!(options[9], 40, "GPU address bits");

    // A direct queue (CreateCommandQueueX, 0x180); the type sits at +0x10.
    let mut queue = null;
    let qdesc = QueueDescX { kind: 0, flags: 0, engine_or_pipe: 0, queue: 0 };
    type CreateQueueX = extern "system" fn(This, *const QueueDescX, *const Iid, *mut This) -> HRESULT;
    assert_eq!(at::<CreateQueueX>(device, 0x180)(device, &qdesc, &QUEUE_IID, &mut queue), S_OK);
    assert_eq!(field::<u32>(queue, 0x10), 0);

    // Allocator (0x48) and list (CreateCommandListX, 0x248) as List6.
    let mut allocator = null;
    type CreateAllocator = extern "system" fn(This, u32, *const Iid, *mut This) -> HRESULT;
    assert_eq!(at::<CreateAllocator>(device, 0x48)(device, 0, &ALLOCATOR_IID, &mut allocator), S_OK);
    let mut list = null;
    let ldesc = ListDesc { kind: 0, flags: 0, nodes: 0, barrier_batch: 0 };
    type CreateListX = extern "system" fn(This, *const ListDesc, This, This, *const Iid, *mut This) -> HRESULT;
    assert_eq!(at::<CreateListX>(device, 0x248)(device, &ldesc, allocator, null, &LIST6_IID, &mut list), S_OK);
    // Put pointer and limits are zero: every inline packet write falls
    // back to the table.
    let (put, draw_limit) = (field::<u64>(list, 0x18), field::<u64>(list, 0x28));
    assert!(put >= draw_limit && put == 0);
    assert_eq!(at::<Count>(list, 0x40)(list), 0, "list type");

    // Fence (0x120); the title reads its value at its GPU address (0x60).
    let mut fence = null;
    type CreateFence = extern "system" fn(This, u64, u32, *const Iid, *mut This) -> HRESULT;
    assert_eq!(at::<CreateFence>(device, 0x120)(device, 0, 0, &FENCE_IID, &mut fence), S_OK);
    let value_at = at::<extern "system" fn(This) -> u64>(fence, 0x60)(fence) as *const u64;

    // A draw (dropped) and a bottom-of-pipe write (newer slot 0x240), close
    // (0x48), submit (0x50): the write lands when the list has run.
    let mut written = 0u64;
    at::<extern "system" fn(This, u32, u32, u32, u32)>(list, 0x60)(list, 3, 1, 0, 0);
    at::<extern "system" fn(This, u64, u64, u32, u32)>(list, 0x240)(list, (&raw mut written) as u64, 42, 0, 0);
    assert_eq!(at::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list), S_OK);
    type Execute = extern "system" fn(This, u32, *const This);
    at::<Execute>(queue, 0x50)(queue, 1, &list);
    // SAFETY: written by the queue after the submission ran.
    assert_eq!(unsafe { (&raw const written).read_volatile() }, 42);

    // Queue signal (newer slot 0x80), then the fence's completed value
    // (0x40), its memory, and a blocking wait (0x48 with no event).
    type Signal = extern "system" fn(This, This, u64) -> HRESULT;
    assert_eq!(at::<Signal>(queue, 0x80)(queue, fence, 1), S_OK);
    assert_eq!(at::<extern "system" fn(This) -> u64>(fence, 0x40)(fence), 1);
    // SAFETY: the fence's value.
    assert_eq!(unsafe { value_at.read_volatile() }, 1);
    type WaitFor = extern "system" fn(This, u64, *mut c_void) -> HRESULT;
    let fence_addr = fence as usize;
    let signaller = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(30));
        let fence = fence_addr as This;
        at::<extern "system" fn(This, u64) -> HRESULT>(fence, 0x50)(fence, 2)
    });
    assert_eq!(at::<WaitFor>(fence, 0x48)(fence, 2, std::ptr::null_mut()), S_OK);
    assert_eq!(signaller.join().unwrap(), S_OK);

    // Reset (0x50), close and submit the empty list again.
    type Reset = extern "system" fn(This, This, This) -> HRESULT;
    assert_eq!(at::<extern "system" fn(This) -> HRESULT>(allocator, 0x40)(allocator), S_OK);
    assert_eq!(at::<Reset>(list, 0x50)(list, allocator, null), S_OK);
    assert_eq!(at::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list), S_OK);
    at::<Execute>(queue, 0x50)(queue, 1, &list);

    // A committed buffer (0xd8): mapped at its GPU address.
    let heap = HeapProperties { kind: 1, cpu_pages: 0, pool: 0, creation_nodes: 1, visible_nodes: 1 };
    let rdesc = ResourceDesc { dimension: BUFFER, width: 1 << 16, height: 1, depth_or_array_size: 1, mip_levels: 1, sample_count: 1, layout: 1, ..Default::default() };
    type CreateCommitted = extern "system" fn(This, *const HeapProperties, u32, *const ResourceDesc, u32, *const c_void, *const Iid, *mut This) -> HRESULT;
    let mut buffer = null;
    assert_eq!(at::<CreateCommitted>(device, 0xd8)(device, &heap, 0, &rdesc, 0, null.cast(), &RESOURCE_IID, &mut buffer), S_OK);
    let gpu = at::<extern "system" fn(This) -> u64>(buffer, 0x58)(buffer);
    let mut mapped: *mut u8 = std::ptr::null_mut();
    assert_eq!(at::<extern "system" fn(This, u32, *const c_void, *mut *mut u8) -> HRESULT>(buffer, 0x40)(buffer, 0, null.cast(), &mut mapped), S_OK);
    assert_eq!(mapped as u64, gpu);
    // SAFETY: 64 KiB were handed out.
    unsafe { mapped.add(0xfff0).write(7) };

    // A shader-visible view heap (0x70): 32-byte descriptors (0x78).
    let hdesc = DescriptorHeapDesc { kind: 0, count: 16, flags: 1, nodes: 0 };
    let placement_none = [hdesc.kind, hdesc.count, hdesc.flags, hdesc.nodes, 0, 0, 0, 0];
    let mut dheap = null;
    type CreateHeap = extern "system" fn(This, *const u32, *const Iid, *mut This) -> HRESULT;
    let heap_iid: Iid = [0x1d, 0x47, 0xfb, 0x8e, 0x6c, 0x61, 0x49, 0x4f, 0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51];
    assert_eq!(at::<CreateHeap>(device, 0x70)(device, placement_none.as_ptr(), &heap_iid, &mut dheap), S_OK);
    assert_ne!(at::<extern "system" fn(This) -> u64>(dheap, 0x48)(dheap), 0);
    assert_eq!(at::<extern "system" fn(This, u32) -> u32>(device, 0x78)(device, 0), 32);

    // A heap in title memory (the wrapper at title+0x1cf3280 passes a base
    // and size after the description, then traps at title+0x1cf3358 unless
    // slot 9 returns that base in RAX).
    let mut title_memory = vec![0u64; 16 * 32 / 8];
    let base = title_memory.as_mut_ptr() as u64;
    let placed = [0u32, 16, 1, 0, base as u32, (base >> 32) as u32, 16 * 32, 0];
    let mut pheap = null;
    assert_eq!(at::<CreateHeap>(device, 0x70)(device, placed.as_ptr(), &heap_iid, &mut pheap), S_OK);
    assert_eq!(at::<extern "system" fn(This) -> u64>(pheap, 0x48)(pheap), base);
    assert_eq!(at::<Count>(pheap, 0x10)(pheap), 0);

    // GetCustomHeapProperties (0xd0) as the title calls it at
    // title+0x247870d: result storage in RCX, the device in RDX; and in
    // the desktop order too.
    type CustomHeap = extern "system" fn(*mut c_void, *mut c_void, u32, u32) -> *mut HeapProperties;
    let mut props = HeapProperties { kind: 0, cpu_pages: 0, pool: 0, creation_nodes: 0, visible_nodes: 0 };
    let custom = at::<CustomHeap>(device, 0xd0);
    let back = custom((&raw mut props).cast(), device.cast_mut().cast(), 1, 2);
    assert_eq!((back, props.kind, props.cpu_pages, props.pool), (&raw mut props, 4, 3, 1));
    let mut props2 = HeapProperties { kind: 0, cpu_pages: 0, pool: 0, creation_nodes: 0, visible_nodes: 0 };
    custom(device.cast_mut().cast(), (&raw mut props2).cast(), 1, 1);
    assert_eq!((props2.kind, props2.cpu_pages), (4, 1));

    // Root signature: serialised by the export, created by the device (0x80).
    type Serialize = extern "system" fn(*const VersionedRootDesc, *mut *mut c_void, *mut *mut c_void) -> HRESULT;
    // SAFETY: the export has this signature.
    let serialize: Serialize = unsafe { std::mem::transmute(export("D3D12SerializeVersionedRootSignature")) };
    let root_desc = VersionedRootDesc {
        version: 1,
        desc: RootDesc { parameter_count: 0, parameters: null.cast(), sampler_count: 0, samplers: null.cast(), flags: 0x0100_0001 },
    };
    let (mut blob, mut error) = (std::ptr::null_mut(), std::ptr::null_mut());
    assert_eq!(serialize(&root_desc, &mut blob, &mut error), S_OK);
    let blob = blob.cast_const().cast();
    let bytes = at::<extern "system" fn(This) -> *const u8>(blob, 0x18)(blob);
    let size = at::<extern "system" fn(This) -> usize>(blob, 0x20)(blob);
    assert!(size > 0);
    let mut root = null;
    let root_iid: Iid = [0x66, 0x6b, 0x4a, 0xc5, 0xdf, 0x72, 0xe8, 0x4e, 0x8b, 0xe5, 0xa9, 0x46, 0xa1, 0x42, 0x92, 0x14];
    type CreateRoot = extern "system" fn(This, u32, *const u8, usize, *const Iid, *mut This) -> HRESULT;
    assert_eq!(at::<CreateRoot>(device, 0x80)(device, 0, bytes, size, &root_iid, &mut root), S_OK);

    // PresentX (newer slot 0x100) three times: logged and paced.
    type Present = extern "system" fn(This, u32, *const u8, *const u8) -> HRESULT;
    let start = Instant::now();
    for _ in 0..3 {
        assert_eq!(at::<Present>(queue, 0x100)(queue, 0, null.cast(), null.cast()), S_OK);
    }
    assert_eq!(d3d12x::presented(), 3);
    // The second and third presents wait for their ticks (60 Hz).
    assert!(start.elapsed() >= Duration::from_millis(15), "presents paced: {:?}", start.elapsed());

    // Everything handed out goes back (0x10 Release returns the count).
    for object in [root, dheap, buffer, list, allocator, fence, queue] {
        assert_eq!(at::<Count>(object, 0x10)(object), 0);
    }
    at::<Count>(blob, 0x10)(blob);
    println!("{}", d3d12x::inventory());
}
