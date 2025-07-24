import subprocess

def run_script(script_name):
    """ 运行指定的 Shell 脚本 """
    try:
        subprocess.run(["bash", script_name], check=True)
        print(f"✅ 成功执行 {script_name}")
    except subprocess.CalledProcessError as e:
        print(f"❌ 运行 {script_name} 失败: {e}")
    except FileNotFoundError:
        print(f"⚠️ 找不到脚本 {script_name}，请检查路径！")

def listen_for_input():
    """ 监听终端输入 """
    print("🎯 监听输入中... 输入 '1' 运行 send+.sh，'2' 运行 send-.sh，'q' 退出")

    while True:
        user_input = input("👉 输入命令: ").strip()

        if user_input == '1':
            run_script("send+.sh")
        elif user_input == '2':
            run_script("send-.sh")
        elif user_input == 'q':
            print("👋 退出程序...")
            break
        else:
            print("⚠️ 无效输入，请输入 '1', '2' 或 'q'")

if __name__ == "__main__":
    listen_for_input()
