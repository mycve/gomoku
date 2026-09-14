use go19::model::PolicyValueModel;
use std::{
    io::Write,
    process::{Command, Stdio},
    time::{SystemTime, UNIX_EPOCH},
};

#[test]
fn executable_speaks_gtp_without_stdout_logs() {
    let path = std::env::temp_dir().join(format!(
        "go9-gtp-{}-{}.safetensors",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    PolicyValueModel::random(8, 81).save(&path).unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_go19"));
    command
        .arg("gtp")
        .arg("--model")
        .arg(&path)
        .args(["--simulations", "4"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }
    let mut child = command.spawn().unwrap();
    child.stdin.take().unwrap().write_all(b"1 protocol_version\n2 boardsize 19\n3 komi 6.5\n4 play b T19\n5 reg_genmove w\n6 genmove w\n7 undo\n8 clear_board\n9 play b pass\n10 play w pass\n11 final_score\n12 quit\n13 name\n").unwrap();
    let result = child.wait_with_output().unwrap();
    std::fs::remove_file(path).unwrap();
    assert!(result.status.success());
    assert!(
        result.stderr.is_empty(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let output = String::from_utf8(result.stdout).unwrap();
    let responses = output.trim_end().split("\n\n").collect::<Vec<_>>();
    assert_eq!(responses.len(), 12, "{output}");
    for (i, response) in responses.iter().enumerate() {
        assert!(response.starts_with(&format!("={}", i + 1)), "{output}");
    }
    assert_eq!(responses[0], "=1 2");
    assert_eq!(responses[10], "=11 W+6.5");
    assert_eq!(
        responses[4].strip_prefix("=5 "),
        responses[5].strip_prefix("=6 ")
    );
}
